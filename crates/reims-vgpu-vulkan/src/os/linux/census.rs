//! A census of the kernel mappings the library's views of memory hold ([`super::budget`]), by
//! the kind of view that holds them: a diagnostic, logged at info as the total rises past each
//! further multiple of [`STEP`] or falls a whole step below the last one logged, and as the first
//! mapping of each kind is made.
//!
//! The kinds ([`Kind`]):
//! - **mapping tasks' views** of the guest's pages: a remap into a mapping task's reservation,
//!   the range `protect_max` takes to no access whole, as only core's `MappingTask` does
//!   ([`protected`]);
//! - **core's other mappings** of them: any other remap through core's Mach VM calls
//!   (`MemoryMap::new_range_mapping`), which a core `Mapping` holds;
//! - **aliases**: the platform's alias of the guest's pages and its zeros, which a heap, a
//!   replace or a buffer over a surface's alias is made over (`crate::gpu::backing::gather`);
//! - **surface views**: an IOSurface stand-in's view of several ranges of the guest's pages
//!   (`super::surface`);
//! - **ranges a view can be made of** (`super::vm::allocate_aliasable`): core's detach
//!   temporaries, an alias with zeros past its pages, the block of zeros;
//! - **import twins** (`super::import`) and **stand-ins' own memory** (`super::vm::OwnMemory`),
//!   which the budget counts apart;
//! - any **other** view.
//!
//! Beside them the census names the aliases parked (`super::vm::park`), each one mapping of no
//! access the table does not count, and the unmaps that wait for a runtime import's hold
//! ([`crate::remap::UnmapHold`]), with the mappings of the runs they keep, whatever their kind.
//! Diagnostics are free under the contract: nothing here changes a mapping or what any call
//! answers.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use reims_vgpu_core::sync::Mutex;

use super::{budget, files};

/// The step of the total the census is logged at.
const STEP: u64 = 50_000;

/// The kind of view a kernel mapping of the library's is held for (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A mapping task's view of the guest's pages.
    Task,
    /// Core's other mappings of the guest's pages.
    Mapping,
    /// The platform's aliases of the guest's pages, and their zeros.
    Alias,
    /// IOSurface stand-ins' views of the guest's pages.
    Surface,
    /// Ranges a view can be made of.
    Range,
    /// Import twins.
    Twin,
    /// IOSurface stand-ins' own memory.
    StandIn,
    /// Any other view.
    Other,
}

impl Kind {
    /// Every kind, in [`Kind::index`] order.
    pub(crate) const ALL: [Kind; 8] = [
        Kind::Task,
        Kind::Mapping,
        Kind::Alias,
        Kind::Surface,
        Kind::Range,
        Kind::Twin,
        Kind::StandIn,
        Kind::Other,
    ];

    /// How many kinds there are.
    pub(crate) const COUNT: usize = Kind::ALL.len();

    /// The kind's place in [`Kind::ALL`].
    pub(crate) fn index(self) -> usize {
        self as usize
    }

    /// What the census line calls it.
    fn name(self) -> &'static str {
        match self {
            Kind::Task => "mapping tasks' views",
            Kind::Mapping => "core's other mappings",
            Kind::Alias => "aliases",
            Kind::Surface => "surface views",
            Kind::Range => "ranges a view can be made of",
            Kind::Twin => "import twins",
            Kind::StandIn => "stand-ins' memory",
            Kind::Other => "other views",
        }
    }
}

thread_local! {
    /// The kind of the views this thread makes now ([`as_kind`]); `None` for core's.
    static MAKING: Cell<Option<Kind>> = const { Cell::new(None) };
}

/// Runs `f` with the views it makes on this thread counted as `kind`'s.
pub(crate) fn as_kind<R>(kind: Kind, f: impl FnOnce() -> R) -> R {
    /// The kind this thread made before, put back as it drops.
    struct Restore(Option<Kind>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MAKING.with(|m| m.set(self.0));
        }
    }
    let _restore = Restore(MAKING.with(|m| m.replace(Some(kind))));
    f()
}

/// The mapping tasks' reservations, by first byte, to their end (module docs). A leaf lock.
static TASKS: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());

/// `length` bytes from `first` were protected `prot` (`PROT_*`): a range of no access that lies
/// in no task's reservation is a new one's.
pub(super) fn protected(first: u64, length: u64, prot: i32) {
    if prot != libc::PROT_NONE {
        return;
    }
    let end = first.saturating_add(length);
    let mut tasks = TASKS.lock();
    let inside = (tasks.range(..=first).next_back()).is_some_and(|(_, &task_end)| end <= task_end);
    if !inside {
        tasks.insert(first, end);
    }
}

/// `length` bytes from `first` were unmapped: the reservations that start there go.
pub(super) fn unmapped(first: u64, length: u64) {
    let end = first.saturating_add(length);
    let mut tasks = TASKS.lock();
    let gone: Vec<u64> = tasks.range(first..end).map(|(&at, _)| at).collect();
    for at in gone {
        tasks.remove(&at);
    }
}

/// The kind of a view this thread makes at `target`: the one [`as_kind`] set, else a mapping
/// task's where `target` lies in one's reservation, else core's other mappings'.
pub(super) fn view_kind(target: u64) -> Kind {
    if let Some(kind) = MAKING.with(Cell::get) {
        return kind;
    }
    let tasks = TASKS.lock();
    match tasks.range(..=target).next_back() {
        Some((_, &end)) if target < end => Kind::Task,
        _ => Kind::Mapping,
    }
}

/// The aliases parked now (`super::vm::park`).
static PARKED: AtomicU64 = AtomicU64::new(0);

/// An alias was parked, or its views made again.
pub(super) fn parked(parked: bool) {
    if parked {
        PARKED.fetch_add(1, Ordering::Relaxed);
    } else {
        PARKED.fetch_sub(1, Ordering::Relaxed);
    }
    changed();
}

/// The mappings of each kind, indexed by [`Kind::index`].
fn counts() -> [u64; Kind::COUNT] {
    let mut counts = files::census();
    counts[Kind::Twin.index()] = budget::twin_mappings();
    counts[Kind::StandIn.index()] = budget::own_mappings();
    counts
}

/// The census line: the mappings of each kind, their total and budget, and the unmaps that wait.
fn line() -> String {
    let counts = counts();
    let kinds: Vec<String> = (Kind::ALL.iter())
        .map(|&k| format!("{} {}", counts[k.index()], k.name()))
        .collect();
    let waiting = crate::remap::waiting_unmaps();
    let bytes: u64 = waiting.iter().map(|w| w.end - w.start).sum();
    let kept: u64 = (waiting.iter())
        .map(|w| files::mappings_between(w.start, w.end))
        .sum();
    format!(
        "views of memory hold {} kernel mappings of a budget of {}: {}; {} aliases parked; {} \
         unmaps of {bytes:#x} bytes wait for imports, keeping {kept} of them",
        counts.iter().sum::<u64>(),
        budget::budget_now(),
        kinds.join(", "),
        PARKED.load(Ordering::Relaxed),
        waiting.len(),
    )
}

/// The first mapping of each kind logged ([`made`]).
static FIRST: [AtomicBool; Kind::COUNT] = [const { AtomicBool::new(false) }; Kind::COUNT];

/// The step of the total last logged ([`changed`]).
static LEVEL: AtomicU64 = AtomicU64::new(0);

/// A view of `kind` was made: the census is logged at the first of its kind, and as
/// [`changed`] logs it. Called under no lock.
pub(super) fn made(kind: Kind) {
    if !FIRST[kind.index()].swap(true, Ordering::Relaxed) {
        log::info!(
            target: "reims_vgpu_vulkan::os",
            "the first of {}: {}",
            kind.name(),
            line()
        );
    }
    changed();
}

/// The views' mappings changed: the census is logged where their total rose past a further
/// multiple of [`STEP`], or fell a whole step below the last one logged (module docs). Called
/// under no lock.
pub(super) fn changed() {
    let level = budget::held() / STEP;
    let last = LEVEL.load(Ordering::Relaxed);
    let moved = level > last || level.saturating_add(1) < last;
    if moved
        && LEVEL
            .compare_exchange(last, level, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        let (way, mark) = match level > last {
            true => ("rose past", level),
            false => ("fell below", level + 1),
        };
        log::info!(
            target: "reims_vgpu_vulkan::os",
            "the views' mappings {way} {}: {}",
            mark * STEP,
            line()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A view's kind is the one its maker set while it runs, the one before after it; else a
    /// mapping task's inside a range protected to no access whole, until it is unmapped, and
    /// core's other mappings' elsewhere
    #[test]
    fn a_views_kind_is_its_makers_or_its_reservations() {
        let base = 0x6e00_0000_0000;
        assert_eq!(view_kind(base), Kind::Mapping);
        protected(base, 0x10_0000, libc::PROT_NONE);
        protected(base + 0x4000, 0x4000, libc::PROT_NONE);
        protected(base + 0x20_0000, 0x4000, libc::PROT_READ);
        assert_eq!(view_kind(base + 0xf_f000), Kind::Task);
        assert_eq!(view_kind(base + 0x10_0000), Kind::Mapping);
        assert_eq!(view_kind(base + 0x20_0000), Kind::Mapping);
        let inner = as_kind(Kind::Alias, || {
            (view_kind(base), as_kind(Kind::Surface, || view_kind(base)))
        });
        assert_eq!(inner, (Kind::Alias, Kind::Surface));
        assert_eq!(view_kind(base), Kind::Task);
        unmapped(base, 0x10_0000);
        assert_eq!(view_kind(base + 0x8000), Kind::Mapping);
    }

    /// The kinds are listed in their index's order
    #[test]
    fn the_kinds_are_in_index_order() {
        for (i, kind) in Kind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i);
        }
    }
}
