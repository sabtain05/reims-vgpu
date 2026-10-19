//! The textures a heap keeps apart ([`Alone`]) in heaps over the guest's pages, against the
//! objects of other heaps, buffers and surfaces over the same pages.
//!
//! A heap over address ranges, core's detached heap, lies over the guest's pages, and core makes
//! other objects over the guest's pages as the guest's records ask: a buffer over pages a heap's
//! record lies over too, a second heap over them, a texture over such a buffer's bytes, a texture
//! on a surface made over them, and a no-copy buffer over a mapping of the guest's pages of its
//! own, a fresh remap of a task's pages (a paged texture's or buffer's transfer buffer, which its
//! paging blits read and write, the XR10 plane views over one, a legacy heap's, a task's shared
//! buffer without detach backing), and textures over that. Core keeps no record of which pages
//! an object holds, and the resources of several tasks over one pool of pages are made (the
//! suite's `native_alias`). Every object but the last kind names the guest's pages by the memory
//! map's host addresses; a mapping names them by its own, so their bytes are named by the memory
//! map's wherever they are compared ([`crate::remap::as_guests`]). A host without a memory map
//! maps a task's pages itself, which this platform does not see, but no heap there lies over
//! address ranges, so nothing is kept apart over them.
//! [`Placements`](crate::gpu::resource::Placements) keeps a heap's own buffers and textures off
//! the bytes of a texture it keeps apart; an object of another heap, over a buffer or on a
//! surface meets those bytes through the guest's pages, where Metal's meets Metal's layout of
//! the texture and the runtime's its own. So, while such a texture lives:
//! - a command's use of bytes of a buffer that lie over the guest's pages under it dies
//!   ([`crate::gpu::resource::BufferObject::used`]), and so does a heap made resident whose
//!   bytes do ([`crate::gpu::resource::HeapObject::made_resident`]);
//! - a texture made over a buffer's bytes, placed in another heap or made on a surface over the
//!   pages under it dies where it is made ([`over_buffer`], [`placed`], [`over_surface`]), and
//!   such a texture placed over pages a texture of any of those kinds lies over, or that a
//!   command used through another heap or a buffer, dies where it is placed ([`placed`]);
//! - a replace that brings a heap's or buffer's bytes, and so the textures in them or over them,
//!   over pages such a texture lies over, or such a texture over pages another texture lies over
//!   or a command used, dies ([`moved`]). Core makes its detached heaps and buffers over memory
//!   of its own and gives them the guest's pages at their first prepare, after the textures in
//!   them are made, and other pages again after a discard.
//!
//! A texture lives here while it or a view of it lives, views of views included, as a Metal
//! view's `parentTexture` keeps the texture and its bytes
//! ([`TextureObject::life`](crate::gpu::resource::TextureObject::life)). Two such textures, or
//! such a texture and the pages a command used, so never lie over one page while both live: a
//! later use of either texture or of a view of it (a bind, an attachment, a copy) meets nothing
//! it would meet on Metal, and needs no check of its own.
//!
//! A command's use of a heap's or buffer's bytes over the guest's pages is kept from the
//! object's replace, which gives it pages, to its detach, which gives them up ([`Seen`]): core
//! detaches an object the guest discards, and the guest may then give its pages to another one.
//! A buffer made over them, on a surface or over core's mapping of them
//! ([`BufferObject::over_guest_pages`](crate::gpu::resource::BufferObject::over_guest_pages)),
//! keeps its uses while it lives and is attached.
//! A detached object's uses fail at their submission, so none is kept while it is detached, over
//! the memory core made it over. A use is kept whether or not a texture is kept apart yet, so such
//! a texture placed later over the bytes a command encoded before reaches dies. One placed in a
//! heap whose cover, core's buffer over all of it, lives dies at its placement
//! ([`Placements`](crate::gpu::resource::Placements)).
//!
//! Each check and the record it keeps are one hold of the registry's lock, so two objects made
//! over one page at once on two threads (two tasks over one pool of pages) never both pass. A use
//! is kept before the textures kept apart are counted, and a texture kept apart is counted before
//! the kept uses are read, both in one order of all threads (`SeqCst`): a use and such a texture
//! made at once over one page meet in one of the two checks.
//!
//! The registry reads an object's pages and uses through its [`Seen`], never through the object:
//! a reference that revived a heap or buffer could run its release, and its hold on Metal's
//! address space with it, on the thread that checks.
//!
//! A buffer made over such pages is no death: a command reaches its bytes only through a use,
//! which dies, and the guest's CPU reaches them through its pages, as it reaches the texture's
//! own, which the texture's notice relays
//! ([`Placements`](crate::gpu::resource::Placements)). No suite case makes one: the suite's
//! textures of those kinds (its `Depth32Float_Stencil8` render targets at offset 0 of a heap of
//! their own) lie in heaps whose pages no other record of the case lies over.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::sync::Mutex;

use crate::answer::plain;
use crate::gpu::resource::{Alone, BufferObject, HeapObject};
use crate::gpu::writes::Memory;

/// A texture's life ([`TextureObject::life`](crate::gpu::resource::TextureObject::life)): alive
/// while its strong count is not 0, which a check reads without reviving it.
pub(crate) type Texture = Weak<dyn std::any::Any + Send + Sync>;

/// What the registry knows of a buffer's or heap's memory, which its
/// [`Backing`](crate::gpu::backing::Backing) keeps and the registry reads without the object: the
/// guest's pages it is over now, and the host memory commands used through it since it was given
/// them.
#[derive(Debug, Default)]
pub struct Seen {
    /// The pages, and whether a detach left the object in no memory since its last replace.
    paged: Mutex<Paged>,
    /// Whether it is over the guest's pages now: `paged` has pages and is not zeroed.
    guests: AtomicBool,
    /// The host memory commands used through it, those that meet merged.
    uses: Mutex<Vec<Range<u64>>>,
    /// Whether `uses` hold all of its pages, so that a use keeps nothing more.
    all_used: AtomicBool,
    /// Whether the registry lists it among those whose uses a check reads.
    listed: AtomicBool,
    /// Moved after each use is kept, and read before `uses` are, in the order of all threads
    /// (module docs).
    stamp: AtomicU64,
    /// Moved as its pages change or its uses are forgotten, so a buffer's whole use kept at an
    /// earlier epoch is kept again ([`used`]).
    epoch: AtomicU64,
}

/// A [`Seen`]'s pages.
#[derive(Debug, Default)]
struct Paged {
    /// The guest's pages, as the host maps them (each a first byte and an end): those of the
    /// object's last replace, or of the address ranges a heap was made over.
    pages: Vec<Range<u64>>,
    /// Whether a detach left the object in no memory since, where the GPU reads zeros through
    /// it.
    zeroed: bool,
}

impl Seen {
    /// The object is over `pages` from now on, its uses forgotten.
    pub(crate) fn set_pages(&self, pages: Vec<Range<u64>>) {
        let guests = !pages.is_empty();
        *self.paged.lock() = Paged {
            pages,
            zeroed: false,
        };
        self.guests.store(guests, Ordering::Release);
        self.forget_uses();
    }

    /// The epoch of its pages and uses.
    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// A detach left the object in no memory: the GPU reads zeros through it, and its writes
    /// reach no guest page.
    pub(crate) fn zero(&self) {
        self.paged.lock().zeroed = true;
        self.guests.store(false, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// The object gave its pages up (a detach): the uses kept are forgotten.
    pub(crate) fn forget_uses(&self) {
        let mut uses = self.uses.lock();
        uses.clear();
        self.all_used.store(false, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// The host memory of the object's bytes `bytes` in the guest's pages it is over now, read
    /// under its lock with no copy of the pages; `None` for one over memory of its own, or
    /// detached.
    pub(crate) fn in_pages(&self, bytes: &Range<u64>) -> Option<Vec<Range<u64>>> {
        let paged = self.paged.lock();
        (!paged.zeroed && !paged.pages.is_empty()).then(|| crate::os::in_runs(&paged.pages, bytes))
    }

    /// Whether a detach left the object in no memory since its last replace.
    pub(crate) fn zeroed(&self) -> bool {
        self.paged.lock().zeroed
    }

    /// Whether the object is over the guest's pages now.
    pub(crate) fn guests(&self) -> bool {
        self.guests.load(Ordering::Acquire)
    }

    /// The host memory of the object's bytes `bytes`: in its pages, none once a detach left it in
    /// no memory; `None` for one never given pages, over memory of its own.
    fn memory(&self, bytes: &Range<u64>) -> Option<Vec<Range<u64>>> {
        let paged = self.paged.lock();
        if paged.zeroed {
            return Some(Vec::new());
        }
        (!paged.pages.is_empty()).then(|| crate::os::in_runs(&paged.pages, bytes))
    }

    /// Keeps a command's use of `memory`, unless a use kept holds all of it already: that one
    /// was kept, and its stamp moved, before this one could see it (module docs).
    fn keep(&self, memory: &[Range<u64>]) {
        {
            let mut uses = self.uses.lock();
            if memory.iter().all(|m| holding(&uses, m)) {
                return;
            }
            for used in memory {
                merge(&mut uses, used.clone());
            }
            let paged = self.paged.lock();
            let pages = &paged.pages;
            if !paged.zeroed && !pages.is_empty() && pages.iter().all(|p| holding(&uses, p)) {
                self.all_used.store(true, Ordering::Release);
            }
        }
        self.stamp.fetch_add(1, Ordering::SeqCst);
    }

    /// The host memory commands used through the object, as kept.
    fn used(&self) -> Vec<Range<u64>> {
        self.stamp.load(Ordering::SeqCst);
        self.uses.lock().clone()
    }
}

/// Merges `merged` into `ranges`, in order of their first bytes, which share no byte and touch
/// none.
fn merge(ranges: &mut Vec<Range<u64>>, mut merged: Range<u64>) {
    ranges.retain(|earlier| {
        let meets = earlier.start <= merged.end && merged.start <= earlier.end;
        if meets {
            merged = merged.start.min(earlier.start)..merged.end.max(earlier.end);
        }
        !meets
    });
    let at = ranges.partition_point(|earlier| earlier.start < merged.start);
    ranges.insert(at, merged);
}

/// Whether one of `ranges` ([`merge`]'s, in order) holds all of `range`.
fn holding(ranges: &[Range<u64>], range: &Range<u64>) -> bool {
    let after = ranges.partition_point(|r| r.start <= range.start);
    after
        .checked_sub(1)
        .and_then(|at| ranges.get(at))
        .is_some_and(|r| range.end <= r.end)
}

/// Where a texture's texels lie: in the bytes `bytes` of the heap or buffer `seen` is of, or,
/// while that one was never given pages, in `fixed`; in `fixed` alone for a texture on a
/// surface, whose memory stays.
#[derive(Clone, Debug)]
pub(crate) struct Site {
    seen: Option<Arc<Seen>>,
    bytes: Range<u64>,
    fixed: Arc<[Range<u64>]>,
}

impl Site {
    /// The bytes `bytes` of `heap`.
    pub(crate) fn of_heap(heap: &HeapObject, bytes: Range<u64>) -> Site {
        Site {
            seen: Some(Arc::clone(&heap.backing.seen)),
            bytes,
            fixed: Arc::new([]),
        }
    }

    /// The bytes `bytes` of `buffer`, where
    /// [`buffer_bytes_memory`](crate::gpu::writes::buffer_bytes_memory) has them: in its
    /// surface's memory, at its place in its heap, in its pages, or in the memory it was made
    /// with.
    pub(crate) fn of_buffer(buffer: &BufferObject, bytes: Range<u64>) -> Site {
        if let Some(s) = &buffer.surface {
            return Site::fixed(Memory::of_surface_bytes(s, bytes).ranges);
        }
        if let Some(h) = &buffer.heap {
            // A heap buffer's GPU address is its heap's plus its offset there.
            let at = plain(|| (buffer.raw.gpu_address().0).saturating_sub(h.raw.gpu_address().0));
            return Site::of_heap(
                h,
                at.saturating_add(bytes.start)..at.saturating_add(bytes.end),
            );
        }
        let contents = plain(|| buffer.raw.contents()).map(|first| {
            let first = crate::os::shown_at(first).as_ptr().addr().to_u64();
            std::iter::once(first.saturating_add(bytes.start)..first.saturating_add(bytes.end))
                .collect()
        });
        Site {
            seen: Some(Arc::clone(&buffer.backing.seen)),
            bytes,
            fixed: crate::remap::as_guests(contents.unwrap_or_default()).into(),
        }
    }

    /// Host memory that stays while the texture lives: a surface's.
    pub(crate) fn fixed(memory: Vec<Range<u64>>) -> Site {
        Site::named(crate::remap::as_guests(memory).into())
    }

    /// [`Site::fixed`], of memory already named as the guest's pages are
    /// ([`crate::remap::as_guests`]).
    pub(crate) fn named(memory: Arc<[Range<u64>]>) -> Site {
        Site {
            seen: None,
            bytes: 0..0,
            fixed: memory,
        }
    }

    /// The host memory the texels lie in now: `fixed` as it is, with no copy, where it is theirs.
    fn memory(&self) -> Cow<'_, [Range<u64>]> {
        match (self.seen.as_ref()).and_then(|seen| seen.memory(&self.bytes)) {
            Some(memory) => Cow::Owned(memory),
            None => Cow::Borrowed(&self.fixed),
        }
    }

    /// Whether the texels lie in the heap or buffer `seen` is of.
    fn of(&self, seen: &Arc<Seen>) -> bool {
        (self.seen.as_ref()).is_some_and(|mine| Arc::ptr_eq(mine, seen))
    }

    /// Whether both lie in one heap or buffer.
    fn with(&self, other: &Site) -> bool {
        other.seen.as_ref().is_some_and(|seen| self.of(seen))
    }
}

/// Whether two memories share a byte.
fn shares(a: &[Range<u64>], b: &[Range<u64>]) -> bool {
    a.iter()
        .any(|a| b.iter().any(|b| a.start < b.end && b.start < a.end))
}

/// What a check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Clash {
    /// Bytes under a texture kept apart, or such a texture over another texture's: the texture
    /// kept apart, and why.
    Pages(Alone),
    /// Such a texture over pages a command used through another heap or a buffer.
    Used(Alone),
}

impl Clash {
    /// The death at `sel`.
    fn dies(self, sel: &'static str) -> ! {
        match self {
            Clash::Pages(why) => why.dies_over_pages(sel),
            Clash::Used(why) => why.dies_over_used_pages(sel),
        }
    }
}

/// A texture kept apart, while it lives.
struct Apart {
    texture: Texture,
    site: Site,
    why: Alone,
}

/// Another texture over a heap's, a buffer's or a surface's bytes, while it lives.
struct Other {
    texture: Texture,
    site: Site,
}

/// The textures kept apart, the others, and the heaps and buffers whose uses are kept, each while
/// it lives (the others and those pruned as they grow).
struct Registry {
    apart: Vec<Apart>,
    others: Vec<Other>,
    /// How many of `others` lived when the rest were last let go.
    kept: usize,
    users: Vec<Weak<Seen>>,
    /// How many of `users` lived when the rest were last let go.
    users_kept: usize,
}

impl Registry {
    /// The first texture kept apart that lives, of another heap than `of`'s, whose texels share
    /// a byte with `memory`.
    fn under(&self, of: Option<&Arc<Seen>>, memory: &[Range<u64>]) -> Result<(), Clash> {
        let theirs = |apart: &&Apart| of.is_none_or(|of| !apart.site.of(of));
        match (self.apart.iter())
            .filter(|apart| apart.texture.strong_count() > 0)
            .filter(theirs)
            .find(|apart| shares(&apart.site.memory(), memory))
        {
            Some(apart) => Err(Clash::Pages(apart.why)),
            None => Ok(()),
        }
    }

    /// A texture kept apart for `why` over `memory` of `site`, against the textures of other
    /// heaps, buffers and surfaces, and the uses of other heaps and buffers.
    fn against(&self, site: &Site, memory: &[Range<u64>], why: Alone) -> Result<(), Clash> {
        let apart = (self.apart.iter())
            .filter(|apart| apart.texture.strong_count() > 0 && !apart.site.with(site))
            .any(|apart| shares(&apart.site.memory(), memory));
        let other = (self.others.iter())
            .filter(|other| other.texture.strong_count() > 0 && !other.site.with(site))
            .any(|other| shares(&other.site.memory(), memory));
        if apart || other {
            return Err(Clash::Pages(why));
        }
        let used = (self.users.iter())
            .filter_map(Weak::upgrade)
            .filter(|user| !site.of(user))
            .any(|user| shares(&user.used(), memory));
        if used {
            return Err(Clash::Used(why));
        }
        Ok(())
    }

    /// Lets go of the textures kept apart that died, and stores how many live in `count`, which
    /// a use then reads: a use checks nothing once none lives.
    fn prune(&mut self, count: &AtomicUsize) {
        let before = self.apart.len();
        self.apart.retain(|apart| apart.texture.strong_count() > 0);
        if self.apart.len() != before {
            count.store(self.apart.len(), Ordering::SeqCst);
        }
    }

    /// Keeps `other`.
    fn keep(&mut self, other: Other) {
        if self.others.len() >= 2 * self.kept.max(16) {
            self.others.retain(|other| other.texture.strong_count() > 0);
            self.kept = self.others.len();
        }
        self.others.push(other);
    }

    /// Lists `user`, whose uses are kept.
    fn list(&mut self, user: &Arc<Seen>) {
        if self.users.len() >= 2 * self.users_kept.max(16) {
            self.users.retain(|user| user.strong_count() > 0);
            self.users_kept = self.users.len();
        }
        self.users.push(Arc::downgrade(user));
    }
}

/// The registry, and how many textures it keeps apart (the dead among them until a check prunes
/// them): a use checks nothing while it is 0.
pub(crate) struct Watch {
    registry: Mutex<Registry>,
    apart: AtomicUsize,
}

/// The process's [`Watch`].
static WATCH: Watch = Watch::new();

impl Watch {
    /// An empty one.
    pub(crate) const fn new() -> Watch {
        Watch {
            registry: Mutex::new(Registry {
                apart: Vec::new(),
                others: Vec::new(),
                kept: 0,
                users: Vec::new(),
                users_kept: 0,
            }),
            apart: AtomicUsize::new(0),
        }
    }

    /// Whether a texture kept apart may live.
    fn any(&self) -> bool {
        self.apart.load(Ordering::SeqCst) > 0
    }

    /// A texture `texture` kept apart for `why`, whose texels lie at `site`: kept unless another
    /// object's texture or use lies over its pages.
    pub(crate) fn apart(&self, site: Site, texture: Texture, why: Alone) -> Result<(), Clash> {
        self.apart_then(site, texture, why, (|| (), || ()))
    }

    /// [`Watch::apart`], running `held` once it holds the registry's lock, before the texture is
    /// counted, and `between` after the check, before the texture is kept.
    fn apart_then(
        &self,
        site: Site,
        texture: Texture,
        why: Alone,
        (held, between): (impl FnOnce(), impl FnOnce()),
    ) -> Result<(), Clash> {
        let mut registry = self.registry.lock();
        held();
        // Counted before the uses are read (module docs).
        self.apart.fetch_add(1, Ordering::SeqCst);
        // Stored once the texture is kept or not, below: a count stored now would leave it out.
        registry
            .apart
            .retain(|apart| apart.texture.strong_count() > 0);
        let clash = registry.against(&site, &site.memory(), why);
        between();
        if clash.is_ok() {
            registry.apart.push(Apart { texture, site, why });
        }
        self.apart.store(registry.apart.len(), Ordering::SeqCst);
        clash
    }

    /// Another texture `texture`, whose texels lie at `site`: kept unless they lie under a
    /// texture another heap keeps apart.
    pub(crate) fn other(&self, site: Site, texture: Texture) -> Result<(), Clash> {
        self.other_then(site, texture, || ())
    }

    /// [`Watch::other`], running `between` after the check, before the texture is kept.
    fn other_then(
        &self,
        site: Site,
        texture: Texture,
        between: impl FnOnce(),
    ) -> Result<(), Clash> {
        let mut registry = self.registry.lock();
        let clash = registry.under(site.seen.as_ref(), &site.memory());
        between();
        if clash.is_ok() {
            registry.keep(Other { texture, site });
        }
        clash
    }

    /// A command's use of `memory` through the heap or buffer `owner` is of, kept where `keep`
    /// says: it may not lie under a texture another heap keeps apart.
    pub(crate) fn used(
        &self,
        owner: &Arc<Seen>,
        keep: bool,
        memory: &[Range<u64>],
    ) -> Result<(), Clash> {
        self.used_then(owner, keep, memory, || ())
    }

    /// [`Watch::used`], running `between` after the use is kept, before the check.
    fn used_then(
        &self,
        owner: &Arc<Seen>,
        keep: bool,
        memory: &[Range<u64>],
        between: impl FnOnce(),
    ) -> Result<(), Clash> {
        if keep {
            // Listed before any thread sees it listed, and so keeps a use a check reads: a use
            // that saw it listed early could miss a texture kept apart meanwhile, whose check
            // read the users before the owner was among them.
            if !owner.listed.load(Ordering::Acquire) {
                let mut registry = self.registry.lock();
                if !owner.listed.load(Ordering::Relaxed) {
                    registry.list(owner);
                    owner.listed.store(true, Ordering::Release);
                }
            }
            // Kept before the textures kept apart are counted (module docs).
            owner.keep(memory);
        }
        between();
        if !self.any() {
            return Ok(());
        }
        let mut registry = self.registry.lock();
        registry.prune(&self.apart);
        registry.under(Some(owner), memory)
    }

    /// The heap or buffer `seen` is of took other pages: no texture it keeps apart may lie over
    /// another object's texture, or over pages another heap's or buffer's commands used, and no
    /// other texture in or over it under a texture another heap keeps apart. Its own uses were
    /// forgotten at the detach a replace follows.
    ///
    /// It always takes the registry's lock, which a texture kept apart holds from its count to
    /// its record, so a replace and such a texture placed at once meet in one of the two checks.
    /// A replace that read the count first, with no lock while it was 0, would rest on a
    /// placement reading the object's pages under the lock the replace wrote them under, as every
    /// site does now; a fence after the write would not stand in for it, as that read is not
    /// sequentially consistent. A replace is no hot path: it is an object's attach.
    pub(crate) fn moved(&self, seen: &Arc<Seen>) -> Result<(), Clash> {
        let mut registry = self.registry.lock();
        registry.prune(&self.apart);
        for apart in registry.apart.iter().filter(|apart| apart.site.of(seen)) {
            registry.against(&apart.site, &apart.site.memory(), apart.why)?;
        }
        let its = |other: &&Other| other.texture.strong_count() > 0 && other.site.of(seen);
        for other in registry.others.iter().filter(its) {
            registry.under(Some(seen), &other.site.memory())?;
        }
        Ok(())
    }
}

/// A command's use `sel` of `memory` through `owner`'s heap or buffer, a heap made resident or a
/// buffer's bytes ([`crate::gpu::writes::buffer_bytes_memory`]; a placed buffer's lie in its
/// heap, whose `owner` it is): kept where `keep` says (the object attached over the guest's
/// pages, or made over them, on a surface or over core's mapping of them), and it dies where
/// they lie under a texture another heap keeps apart that lives, naming why. `memory`'s bytes
/// in core's mapping of the guest's pages are named by the memory map's addresses
/// ([`crate::remap::as_guests`]). `whole` is a buffer's record of its uses of all of its
/// bytes (`BufferObject::kept_whole`), for such a use. `memory` is read only where needed: a use
/// keeps nothing once all of the object's pages are kept, nor a use of all of a buffer kept
/// since the object's pages last changed, and checks nothing while no texture is kept apart.
pub(crate) fn used(
    sel: &'static str,
    owner: &Arc<Seen>,
    keep: bool,
    whole: Option<&AtomicU64>,
    memory: impl FnOnce() -> Vec<Range<u64>>,
) {
    // Read before the use is kept: a change of pages meanwhile keeps the next use again.
    let epoch = owner.epoch().wrapping_add(1);
    let kept = whole.is_some_and(|whole| whole.load(Ordering::Acquire) == epoch);
    let keep = keep && !kept && !owner.all_used.load(Ordering::Acquire);
    if !keep && !WATCH.any() {
        return;
    }
    let memory = crate::remap::as_guests(memory());
    if memory.is_empty() {
        return;
    }
    if let Err(clash) = WATCH.used(owner, keep, &memory) {
        clash.dies(sel);
    }
    if let Some(whole) = whole.filter(|_| keep) {
        whole.store(epoch, Ordering::Release);
    }
}

/// A texture `texture`, made by `sel` over the bytes `bytes` of `buffer`: it dies where they lie
/// over the guest's pages under a texture a heap other than the buffer's keeps apart that lives,
/// and is kept, so such a texture placed or brought over its pages later dies ([`placed`],
/// [`moved`]).
pub(crate) fn over_buffer(
    sel: &'static str,
    buffer: &BufferObject,
    bytes: Range<u64>,
    texture: Texture,
) {
    if let Err(clash) = WATCH.other(Site::of_buffer(buffer, bytes), texture) {
        clash.dies(sel);
    }
}

/// A texture `texture`, made by `sel` on a surface made over the guest's pages, whose texels lie
/// in `memory`, named as the guest's pages are ([`crate::remap::as_guests`]): as
/// [`over_buffer`]'s.
pub(crate) fn over_surface(sel: &'static str, memory: Arc<[Range<u64>]>, texture: Texture) {
    if let Err(clash) = WATCH.other(Site::named(memory), texture) {
        clash.dies(sel);
    }
}

/// A texture `texture` placed by `sel` over the bytes `bytes` of `heap`, a heap over address
/// ranges, kept apart where `alone` says why: one kept apart dies where those bytes lie over the
/// guest's pages that a texture of another heap kept apart, or another heap's, a buffer's or a
/// surface's texture, lies over and lives, or that a command used through another heap or a
/// buffer; any other dies where they lie under a texture another heap keeps apart. Each is kept,
/// for the textures placed, made or brought over its pages later.
pub(crate) fn placed(
    sel: &'static str,
    heap: &HeapObject,
    bytes: Range<u64>,
    texture: Texture,
    alone: Option<Alone>,
) {
    let site = Site::of_heap(heap, bytes);
    let clash = match alone {
        Some(why) => WATCH.apart(site, texture, why),
        None => WATCH.other(site, texture),
    };
    if let Err(clash) = clash {
        clash.dies(sel);
    }
}

/// `sel`, a replace, gave the heap or buffer `seen` is of other pages: it dies where that brings
/// a texture kept apart and another object's texture, or pages another object's commands used,
/// over one page.
pub(crate) fn moved(sel: &'static str, seen: &Arc<Seen>) {
    if let Err(clash) = WATCH.moved(seen) {
        clash.dies(sel);
    }
}

#[cfg(test)]
#[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
mod tests {
    use super::*;

    /// A stand-in heap or buffer over `pages`.
    fn seen(pages: &[Range<u64>]) -> Arc<Seen> {
        let seen = Arc::new(Seen::default());
        seen.set_pages(pages.to_vec());
        seen
    }

    /// A stand-in texture, and its identity.
    fn texture() -> (Arc<()>, Texture) {
        let texture = Arc::new(());
        let dynamic: Arc<dyn std::any::Any + Send + Sync> = Arc::clone(&texture) as _;
        (texture, Arc::downgrade(&dynamic))
    }

    fn site(seen: &Arc<Seen>, bytes: Range<u64>) -> Site {
        Site {
            seen: Some(Arc::clone(seen)),
            bytes,
            fixed: Arc::new([]),
        }
    }

    const PAGE: u64 = 0x4000;

    /// A texture kept apart in a heap over memory of its own, as core makes its detached heaps,
    /// meets nothing until the replace that gives it the guest's pages, where a texture over a
    /// buffer, another heap's texture, a surface's texture and another heap's kept use lie: each
    /// kills the replace's check. A texture of its own heap, and a use of it, do not; nor does a
    /// texture of another object once it is let go, nor a use forgotten at its object's detach
    #[test]
    fn a_replace_meets_what_lies_over_the_new_pages() {
        let guest = 0x10_0000..0x10_0000 + PAGE;
        for case in ["texture", "surface", "use"] {
            let watch = Watch::new();
            let heap = seen(&[0x20_0000..0x20_0000 + PAGE]);
            let (kept, apart) = texture();
            let why = Alone::Untiled;
            assert_eq!(watch.apart(site(&heap, 0..PAGE), apart, why), Ok(()));
            let (own, mine) = texture();
            assert_eq!(watch.other(site(&heap, 0..PAGE), mine), Ok(()));
            assert_eq!(watch.used(&heap, true, &[0x20_0000..0x20_1000]), Ok(()));
            let other = seen(std::slice::from_ref(&guest));
            let (theirs, weak) = texture();
            let expected = match case {
                "texture" => {
                    watch.other(site(&other, 0..0x100), weak).unwrap();
                    Clash::Pages(why)
                }
                "surface" => {
                    watch.other(Site::fixed(vec![guest.clone()]), weak).unwrap();
                    Clash::Pages(why)
                }
                _ => {
                    watch
                        .used(&other, true, &[guest.start..guest.start + 4])
                        .unwrap();
                    Clash::Used(why)
                }
            };
            assert_eq!(watch.moved(&heap), Ok(()), "{case}: before the replace");
            heap.set_pages(vec![guest.clone()]);
            assert_eq!(watch.moved(&heap), Err(expected), "{case}");
            drop(theirs);
            other.forget_uses();
            assert_eq!(watch.moved(&heap), Ok(()), "{case}: let go");
            drop((kept, own));
        }
    }

    /// A buffer whose texture lies over pages of its own, given a texture kept apart's pages by a
    /// replace, kills the replace's check; given others, it does not
    #[test]
    fn a_replace_brings_a_buffers_texture_under_a_texture_kept_apart() {
        let guest = 0x40_0000..0x40_0000 + PAGE;
        let watch = Watch::new();
        let heap = seen(std::slice::from_ref(&guest));
        let (_t, apart) = texture();
        assert_eq!(
            watch.apart(site(&heap, 0..PAGE), apart, Alone::Untiled),
            Ok(())
        );
        let buffer = seen(&[0x50_0000..0x50_0000 + PAGE]);
        let (_o, other) = texture();
        assert_eq!(watch.other(site(&buffer, 0..0x800), other), Ok(()));
        buffer.set_pages(vec![0x60_0000..0x60_0000 + PAGE]);
        assert_eq!(watch.moved(&buffer), Ok(()));
        buffer.set_pages(vec![guest.clone()]);
        assert_eq!(watch.moved(&buffer), Err(Clash::Pages(Alone::Untiled)));
    }

    /// A use of all of a buffer is kept once while its object's pages stay: a later one reads no
    /// memory, until the object's uses are forgotten or its pages change. A use of part of it is
    /// kept each time
    #[test]
    fn a_whole_use_is_kept_once_while_the_pages_stay() {
        let owner = seen(&[0x70_0000..0x70_0000 + 2 * PAGE]);
        let whole = AtomicU64::new(0);
        let reads = std::cell::Cell::new(0);
        let use_all = || {
            used("a use", &owner, true, Some(&whole), || {
                reads.set(reads.get() + 1);
                vec![0x70_0000..0x70_0000 + PAGE]
            });
        };
        use_all();
        use_all();
        assert_eq!(reads.get(), 1, "kept once");
        assert_eq!(owner.used(), [0x70_0000..0x70_0000 + PAGE]);
        owner.forget_uses();
        use_all();
        assert_eq!(reads.get(), 2, "after the uses are forgotten");
        owner.set_pages(vec![0x80_0000..0x80_0000 + 2 * PAGE]);
        use_all();
        assert_eq!(reads.get(), 3, "after the pages change");
        for _ in 0..2 {
            used("a use", &owner, true, None, || {
                reads.set(reads.get() + 1);
                vec![0x80_0000..0x80_0010]
            });
        }
        assert_eq!(reads.get(), 5, "a use of part of it");
    }

    /// A use that a use kept already holds keeps nothing more, its stamp unmoved; the uses kept
    /// stay merged in order of their first bytes; and a use's check lets go of the textures kept
    /// apart that died, so the next use checks nothing
    #[test]
    fn uses_kept_already_and_dead_textures_cost_nothing_more() {
        let page = 0x300_0000;
        let watch = Watch::new();
        let owner = seen(&[page..page + 4 * PAGE]);
        let stamp = || owner.stamp.load(Ordering::SeqCst);
        let at = |a: u64, b: u64| [page + a..page + b];
        assert_eq!(watch.used(&owner, true, &at(0x2000, 0x3000)), Ok(()));
        assert_eq!(stamp(), 1);
        assert_eq!(watch.used(&owner, true, &at(0x2100, 0x2200)), Ok(()));
        assert_eq!(stamp(), 1, "held already");
        assert_eq!(watch.used(&owner, true, &at(0x100, 0x200)), Ok(()));
        assert_eq!(watch.used(&owner, true, &at(0x3000, 0x3100)), Ok(()));
        assert_eq!(stamp(), 3);
        assert_eq!(
            owner.used(),
            [page + 0x100..page + 0x200, page + 0x2000..page + 0x3100]
        );
        assert!(!owner.all_used.load(Ordering::Acquire));
        assert_eq!(watch.used(&owner, true, &at(0, 4 * PAGE)), Ok(()));
        assert!(owner.all_used.load(Ordering::Acquire), "all of its pages");
        let heap = seen(&[page + 8 * PAGE..page + 9 * PAGE]);
        let (kept, apart) = texture();
        assert_eq!(
            watch.apart(site(&heap, 0..PAGE), apart, Alone::Untiled),
            Ok(())
        );
        assert!(watch.any());
        drop(kept);
        assert!(watch.any(), "counted until a check lets it go");
        assert_eq!(watch.used(&owner, false, &at(0, 0x10)), Ok(()));
        assert!(!watch.any(), "let go at a use's check");
    }

    /// A texture kept apart placed over pages another heap's or buffer's commands used dies, and
    /// a use kept before any texture is kept apart counts; once that object is detached, its uses
    /// are forgotten and the placement is made. A use of all of an object's pages keeps nothing
    /// more after it
    #[test]
    fn a_texture_kept_apart_over_used_pages_dies() {
        let guest = 0x30_0000..0x30_0000 + 2 * PAGE;
        let watch = Watch::new();
        let buffer = seen(std::slice::from_ref(&guest));
        assert_eq!(
            watch.used(&buffer, true, &[guest.start..guest.start + 16]),
            Ok(())
        );
        assert!(!buffer.all_used.load(Ordering::Acquire));
        let heap = seen(std::slice::from_ref(&guest));
        let (_t, apart) = texture();
        let why = Alone::Optimized;
        assert_eq!(
            watch.apart(site(&heap, 0..PAGE), apart.clone(), why),
            Err(Clash::Used(why))
        );
        assert_eq!(
            watch.apart(site(&heap, PAGE..2 * PAGE), apart.clone(), why),
            Ok(()),
            "beside the use"
        );
        buffer.forget_uses();
        assert_eq!(watch.apart(site(&heap, 0..PAGE), apart, why), Ok(()));
        assert_eq!(
            watch.used(&buffer, true, std::slice::from_ref(&guest)),
            Err(Clash::Pages(why))
        );
        assert!(buffer.all_used.load(Ordering::Acquire), "all of its pages");
    }

    /// Core's own mapping of the guest's pages, a no-copy buffer's memory as core makes its
    /// transfer buffers, is named by the guest's pages ([`crate::remap::as_guests`]), so a texture
    /// over such a buffer, and a use of its bytes, meet a texture kept apart over the same pages,
    /// and a use kept there kills such a texture placed later; once the mapping is gone, its
    /// address names its own memory again
    #[test]
    fn memory_in_cores_mapping_of_the_guests_pages_meets_them() {
        use reims_vgpu_core::platform::os::vm::HostVm as _;
        let vm = crate::remap::VulkanVm::new();
        let guest = crate::os::allocate_aliasable(PAGE).unwrap();
        let mapping = vm.allocate_anywhere(PAGE).unwrap();
        vm.remap_shared_overwrite(mapping, guest, PAGE).unwrap();
        let (g, m) = (guest.addr().to_u64(), mapping.addr().to_u64());
        let watch = Watch::new();
        let heap = seen(&[g..g + PAGE]);
        let why = Alone::Untiled;
        let (kept, apart) = texture();
        assert_eq!(watch.apart(site(&heap, 0..PAGE), apart, why), Ok(()));
        let (_o, other) = texture();
        assert_eq!(
            watch.other(Site::fixed(vec![m + 0x100..m + 0x200]), other),
            Err(Clash::Pages(why)),
            "a texture over the mapping"
        );
        let buffer = Arc::new(Seen::default());
        let used = crate::remap::as_guests(vec![m + 0x10..m + 0x20]);
        assert_eq!(used, [g + 0x10..g + 0x20]);
        assert_eq!(
            watch.used(&buffer, true, &used),
            Err(Clash::Pages(why)),
            "a use of the mapping"
        );
        drop(kept);
        let second = seen(&[g..g + PAGE]);
        let (_s, later) = texture();
        assert_eq!(
            watch.apart(site(&second, 0..PAGE), later, why),
            Err(Clash::Used(why)),
            "a texture kept apart over the used pages"
        );
        assert!(vm.deallocate(mapping, PAGE).is_success());
        let (_n, over) = texture();
        assert_eq!(
            watch.other(Site::fixed(vec![m + 0x100..m + 0x200]), over),
            Ok(()),
            "once the mapping is gone"
        );
        let _ = crate::os::OsVm::new().deallocate(guest, PAGE);
    }

    /// Two objects made over one page at once on two threads: the first pauses between its check
    /// and what it keeps, and the second is made meanwhile. Exactly one of the two dies, whichever
    /// pauses: a texture kept apart and a texture over a buffer, either first; a texture kept
    /// apart, then a use; and a use, kept before its pause, then a texture kept apart. Each check
    /// and its record are one hold of the registry's lock, and a use is kept before the textures
    /// kept apart are counted
    #[test]
    fn objects_made_at_once_over_one_page_meet() {
        let pause = || std::thread::sleep(std::time::Duration::from_millis(30));
        for (case, first) in ["apart", "other", "apart before a use", "use"]
            .iter()
            .enumerate()
        {
            let page = 0x100_0000 + u64::try_from(case).unwrap() * PAGE;
            let guest = page..page + PAGE;
            let watch = Watch::new();
            let (heap, buffer) = (
                seen(std::slice::from_ref(&guest)),
                seen(std::slice::from_ref(&guest)),
            );
            let (_a, apart) = texture();
            let (_o, other) = texture();
            let started = std::sync::Barrier::new(2);
            let (one, two) = std::thread::scope(|s| {
                let second = s.spawn(|| {
                    started.wait();
                    match *first {
                        "apart" => watch.other(site(&buffer, 0..PAGE), other.clone()),
                        "apart before a use" => {
                            watch.used(&buffer, true, std::slice::from_ref(&guest))
                        }
                        _ => watch.apart(site(&heap, 0..PAGE), apart.clone(), Alone::Untiled),
                    }
                });
                let between = || {
                    started.wait();
                    pause();
                };
                let one = match *first {
                    "other" => watch.other_then(site(&buffer, 0..PAGE), other.clone(), between),
                    "use" => watch.used_then(&buffer, true, std::slice::from_ref(&guest), between),
                    _ => watch.apart_then(
                        site(&heap, 0..PAGE),
                        apart.clone(),
                        Alone::Untiled,
                        (|| (), between),
                    ),
                };
                (one, second.join().unwrap())
            });
            assert!(one.is_err() != two.is_err(), "{first}: {one:?} {two:?}");
        }
    }

    /// Two uses through one heap or buffer on two threads while a texture kept apart is placed
    /// on a third, which holds the registry's lock before counting it: the first use, of bytes
    /// beside the texture's page, waits to list the object, and the second, over the page,
    /// starts meanwhile. The second waits too, as the object is listed by none yet, so it meets
    /// the texture; it never sees the object listed before it is, keeps its use and checks
    /// before the texture is counted, while the texture's check misses the unlisted object
    #[test]
    fn a_use_never_sees_its_object_listed_before_it_is() {
        let page = 0x200_0000;
        let (beside, guest) = (page + PAGE..page + 2 * PAGE, page..page + PAGE);
        let watch = Watch::new();
        let (heap, buffer) = (
            seen(std::slice::from_ref(&guest)),
            seen(&[page..page + 2 * PAGE]),
        );
        let (_a, apart) = texture();
        let started = std::sync::Barrier::new(3);
        let pause = |ms| std::thread::sleep(std::time::Duration::from_millis(ms));
        let (placed, first, second) = std::thread::scope(|s| {
            let first = s.spawn(|| {
                started.wait();
                watch.used(&buffer, true, std::slice::from_ref(&beside))
            });
            let second = s.spawn(|| {
                started.wait();
                pause(10);
                watch.used(&buffer, true, std::slice::from_ref(&guest))
            });
            let held = || {
                started.wait();
                pause(40);
            };
            let site = site(&heap, 0..PAGE);
            let placed = watch.apart_then(site, apart, Alone::Untiled, (held, || ()));
            (placed, first.join().unwrap(), second.join().unwrap())
        });
        assert_eq!(first, Ok(()), "beside the page");
        assert!(placed.is_err() != second.is_err(), "{placed:?} {second:?}");
    }

    /// Prints how long a texture made on a surface over the guest's pages takes to be named and
    /// kept here, as `newTextureWithDescriptor:iosurface:plane:` has it, while no texture is kept
    /// apart: its memory ([`crate::os::surface_bytes_memory`]), its site and its keeping
    /// ([`over_surface`]'s); for a surface made over 2,025 runs of pages, as an x86 guest's
    /// 1920 × 1080 surface is, beside 2,000 runs of core's mappings
    #[test]
    #[ignore = "a measurement, run by hand: prints the time a call"]
    fn a_texture_on_a_surface_over_2025_runs() {
        use reims_vgpu_core::platform::os::surface::Surface as _;

        let (s, pages, length) = crate::os::tests::bgra_surface_over_runs(2025);
        crate::remap::guest_mappings_for_measurements(0x7000_0000_0000, 2000, 0x10_0000_0000);
        let watch = Watch::new();
        let bytes = [0..s.alloc_size()];
        let rounds = crate::os::tests::microseconds_a_call(200, || {
            let memory = crate::os::surface_bytes_memory(&s, &bytes);
            let (t, life) = texture();
            assert_eq!(watch.other(Site::fixed(memory), life), Ok(()));
            drop(t);
        });
        println!("a texture on a surface over 2,025 runs, µs a call: {rounds:.1?}");
        drop(s);
        crate::os::tests::free(pages, length);
    }
}
