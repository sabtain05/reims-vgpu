//! What this platform reads of a surface beyond core's `Surface` trait, written against it and
//! each OS's record of the surfaces it made (`recorded_mut`): a plane's layout and YCbCr facts,
//! the host memory a surface's bytes are over, whether the GPU's writes through it reach that
//! memory, and the alias its buffers lie over.

use std::sync::{Arc, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::surface::Surface;

use super::{OsSurface, host_os};
use crate::gpu::backing::Alias;

/// Plane `plane` of `surface`, as a texture on it reads it: its bytes' offset from the surface's
/// base address, its bytes per row, and whether the surface has planes. A surface without planes
/// is its plane 0. `None` for a plane the surface lacks, of which Metal makes no texture
/// (`textures_backing_records`). The plane names no `kIOSurfaceClientDirection`, which the
/// runtime refuses: where the GPU's writes through a surface never reach its memory
/// ([`gpu_read_only`]), the buffer under the texture lies over a read-only alias of that memory
/// (the device's `buffer_over_surface_at`), and the GPU's writes never reach it there either.
pub(crate) fn plane(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfacePlane> {
    if surface.plane_count() == 0 {
        return (plane == 0).then(|| rt::IoSurfacePlane::new(0, surface.bytes_per_row(), false));
    }
    let (offset, bytes_per_row) = host_os::plane_at(surface, plane)?;
    Some(rt::IoSurfacePlane::new(offset, bytes_per_row, true))
}

/// What a texture over plane `plane` of `surface`, a surface of two planes, reads of its layout,
/// which Metal reads from it when the texture is made: its OSType, its size, and its plane 1's
/// offset from its base address and bytes per row. `None` for a surface of fewer than two planes,
/// over which the runtime then refuses the texture. A texture of a two-plane XR10 format (550,
/// 551) takes these alone, plane 1 its alpha bytes ([`crate::gpu::device`]).
pub(crate) fn plane_facts(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfaceYCbCr> {
    if surface.plane_count() < 2 {
        return None;
    }
    let (chroma, bytes_per_row) = host_os::plane_at(surface, 1)?;
    Some(rt::IoSurfaceYCbCr::new(
        surface.pixel_format(),
        plane,
        [surface.width(), surface.height()],
        chroma,
        bytes_per_row,
    ))
}

/// What a texture of a two-plane YCbCr format the runtime reads
/// ([`crate::gpu::ycbcr::runtime_makes`]) over plane `plane` of `surface` reads of the surface:
/// [`plane_facts`], and its YCbCr matrix as it is now, where it names one
/// (`kIOSurfaceYCbCrMatrix`).
pub(crate) fn ycbcr_facts(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfaceYCbCr> {
    let facts = plane_facts(surface, plane)?;
    Some(match host_os::ycbcr_matrix(surface) {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    })
}

/// The host memory a surface's bytes are over: its allocation in its own mapping (its
/// allocation size from its base address less its `kIOSurfaceOffset`), and, for one
/// `OsSurfaces::create` made over the guest's pages (`kIOSurfaceClientAddressRanges`),
/// those pages as the host maps them, where the guest's other objects over the same pages lie.
/// Each range is a first byte and an end.
///
/// The surface's base address is its offset into its allocation, and a plane's
/// (`IOSurfaceGetBaseAddressOfPlane`) its `kIOSurfacePlaneOffset` into it. A surface over one
/// run of the guest's pages at offset 0 has them as its base address; at another offset, its
/// allocation is another mapping of the pages, the base address that offset into it, and a
/// write through it shows in the pages at that offset (measured on the M3 Max, offsets 0x40,
/// 0x1000 and 0x4000).
#[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
pub(crate) fn surface_memory(surface: &OsSurface) -> Vec<std::ops::Range<u64>> {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    // The pages copied once, under the record's lock; no record is no pages and offset 0. The
    // surface's own range comes first, its size read after the lock.
    let (offset, mut memory) = recorded(surface, |made| {
        let mut memory = Vec::with_capacity(1 + made.pages.len());
        memory.push(0..0);
        memory.extend_from_slice(&made.pages);
        (made.offset, memory)
    })
    .unwrap_or_else(|| (0, vec![0..0]));
    let first = base.saturating_sub(offset);
    memory[0] = first..first.saturating_add(surface.alloc_size());
    memory
}

/// The host memory of `surface`'s bytes at `offsets`, each a first byte and an end counted from
/// the surface's base address, where [`surface_memory`] has them: from its base address, then in
/// the guest's pages it was made over, in their order, from its offset into them.
pub(crate) fn surface_bytes_memory(
    surface: &OsSurface,
    offsets: &[std::ops::Range<u64>],
) -> Vec<std::ops::Range<u64>> {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    let own = || {
        offsets
            .iter()
            .map(|o| base.saturating_add(o.start)..base.saturating_add(o.end))
    };
    // Read under the record's lock, without copying its pages: a surface over the guest's pages
    // may have a run for each (2,025 for an x86-64 guest's 1080p surface), and a texture on it
    // reads this at each encode that binds or uses it. No record is no pages and offset 0.
    recorded(surface, |made| {
        let at = |o: &std::ops::Range<u64>| {
            made.offset.saturating_add(o.start)..made.offset.saturating_add(o.end)
        };
        let mut memory = Vec::with_capacity(offsets.len() + made.pages.len());
        memory.extend(own());
        for o in offsets {
            each_in_runs(&made.pages, &at(o), |held| memory.push(held));
        }
        memory
    })
    .unwrap_or_else(|| own().collect())
}

/// The host ranges that hold the bytes `offsets` of memory laid out over `runs` in their order
/// (each a host first byte and an end).
pub(crate) fn in_runs(
    runs: &[std::ops::Range<u64>],
    offsets: &std::ops::Range<u64>,
) -> Vec<std::ops::Range<u64>> {
    let mut held = Vec::new();
    each_in_runs(runs, offsets, |range| held.push(range));
    held
}

/// [`in_runs`], each range given to `each` in its order.
fn each_in_runs(
    runs: &[std::ops::Range<u64>],
    offsets: &std::ops::Range<u64>,
    mut each: impl FnMut(std::ops::Range<u64>),
) {
    let mut at = 0u64;
    for run in runs {
        let length = run.end - run.start;
        let (first, end) = (offsets.start.max(at), offsets.end.min(at + length));
        if first < end {
            each(run.start + (first - at)..run.start + (end - at));
        }
        at += length;
        if at >= offsets.end {
            break;
        }
    }
}

/// The offset of `surface`'s bytes into its allocation (`kIOSurfaceOffset`), as
/// `OsSurfaces::create` made it: its base address lies that far into the allocation, a
/// page-aligned mapping of `IOSurfaceGetAllocSize` bytes (measured on the M3 Max at 0x40, 0x1000
/// and 0x4000, over the guest's pages); 0 for a surface made otherwise.
pub(crate) fn surface_offset(surface: &OsSurface) -> u64 {
    recorded(surface, |made| made.offset).unwrap_or_default()
}

/// Whether `surface` was made over the guest's pages (`OsSurfaces::create`).
pub(crate) fn over_guest_pages(surface: &OsSurface) -> bool {
    recorded(surface, |made| !made.pages.is_empty()).unwrap_or(false)
}

/// The `kIOSurfaceClientDirection` of a surface core makes over the guest's pages read-only for
/// its clients (core's `iosurface::descriptor`).
pub(crate) const READ_ONLY_DIRECTION: u32 = 2;

/// Whether the GPU's writes through `surface` never reach its memory: it was made over the
/// guest's pages with `kIOSurfaceClientDirection` 2, which Metal maps read-only for the GPU.
///
/// Measured on the M3 Max (macOS 26.7, one case per process, BGRA8Unorm surfaces of 64 by 16, 256
/// by 256 and 1024 by 1024 texels, `tests/client_direction.rs`): through a texture on such a
/// surface, a blit's copy or colour fill and a kernel's writes of one or two texels reach the later
/// commands of their command buffer and never the pages, which later command buffers and the CPU
/// read unchanged; of a row of 64 texels or more, they fail the command buffer with a GPU page
/// fault (`MTLCommandBufferErrorPageFault`), the encoders after them running nothing. So does a
/// blit's copy of a texture of a row of 64 texels into a buffer made on the surface with
/// `newBufferWithIOSurface:`, where one of two texels is dropped as above. A blit's copy from a
/// buffer or colour fill of 1 to 8 texels each way into the texture is dropped at every origin
/// tried (0,0, 1,1 and one far inside, on surfaces of 64 by 16 and 256 by 256 texels); of a row of
/// 16 to 63 texels it faults at some origins and is dropped at others (16 at 0,0 faults, at 1,1 is
/// dropped). A blit's fill of a buffer made on the surface, its copy into one from another buffer,
/// a kernel's stores into one, plain or by an atomic exchange, and a vertex or fragment function's,
/// of 8 bytes to 4 MiB, are dropped as above, the later commands of the command buffer reading what
/// of the write the GPU's caches still hold (all of 4 MiB with nothing else on the GPU, less beside
/// other GPU work, none after 64 MiB of it; a draw's, all, part or none of it, case to case, in one
/// process). A render pass that clears it or draws into it and stores fails its command buffer so;
/// one that stores nothing, or loads, stores and draws nothing, completes, writing nothing.
/// `replaceRegion:` writes through the surface's base address: the page of a surface over one page,
/// which lies there; a surface over several (4, 16 and 256) has a mapping of its own there,
/// read-only, and the write ends the process with a bus error, at every extent measured
/// ([`base_in_pages`]). The key is not read back (`IOSurfaceCopyValue` answers none), so the
/// surfaces made here record it. A surface of its own memory takes the key and the GPU's writes
/// alike; one over the guest's pages takes 3 and the writes land, and none of 0, 1, 4 or 255
/// (`IOSurfaceCreate` answers NULL); core sets 2 alone.
pub(crate) fn gpu_read_only(surface: &OsSurface) -> bool {
    recorded(surface, |made| made.gpu_read_only).unwrap_or(false)
}

/// Dies at `sel`, a command that may write `what` into the memory of a surface whose GPU writes
/// never reach it ([`gpu_read_only`]): Metal's GPU drops some such writes and faults the command
/// buffer at others by their extent, which nothing here can tell before the GPU runs the
/// command, and the runtime's device loses itself at the same fault, so the write is never sent.
/// A blit's fill of a buffer made on the surface, its copy into one from another buffer, and a
/// kernel's, vertex function's or fragment function's stores into one, which Metal drops at
/// every extent measured, as the runtime does, are sent; a tile function's, which nothing
/// measured, are not. So is a blit's copy from a buffer or colour fill into a `BGRA8Unorm` 2D
/// texture made on the surface, of 1 to 8 texels each way inside level 0, on a surface whose rows
/// hold 256 bytes or more, which Metal drops at every origin measured
/// (`gpu::encode_blit::dropped_where_measured`).
pub(crate) fn dropped_write(sel: &'static str, what: &str) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} a surface made read-only for the GPU (kIOSurfaceClientDirection 2), whose \
             writes Metal's GPU drops or faults on by their extent"
        ),
    )
}

/// The alias of the allocation of `surface`, one at an offset into it ([`surface_offset`]) or
/// whose GPU writes never reach its memory, read-only then ([`gpu_read_only`]), that every buffer
/// over it lies over while one lives: the live one, or the one `make` makes, kept for the next;
/// `None` where `make` makes none. The buffers then lie in one import of the runtime's, as those
/// over a surface at no offset whose writes land lie over its base address: a write through one
/// reaches the later commands of its command buffer that read another, as on Metal
/// (`tests/client_direction.rs`), and a buffer over the surface can share the range of another
/// over the same first byte (the device's `new_buffer_on_surface`).
pub(crate) fn surface_alias(
    surface: &OsSurface,
    make: impl FnOnce() -> Option<Alias>,
) -> Option<Arc<Alias>> {
    let mut make = Some(make);
    let kept = host_os::recorded_mut(surface, |made| {
        if let Some(alias) = made.alias.upgrade() {
            return Some(alias);
        }
        let alias = Arc::new(make.take()?()?);
        made.alias = Arc::downgrade(&alias);
        Some(alias)
    });
    match kept {
        Some(alias) => alias,
        None => make.take()?().map(Arc::new),
    }
}

/// Whether `surface`'s base address lies in the guest's pages it was made over, so that the CPU's
/// writes through it reach them: as for a surface over one page; one over several, contiguous or
/// not, has a mapping of its own (measured on the M3 Max: one of 4 contiguous pages), read-only
/// for a surface of [`gpu_read_only`].
pub(crate) fn base_in_pages(surface: &OsSurface) -> bool {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    recorded(surface, |made| {
        made.pages.iter().any(|run| run.contains(&base))
    })
    .unwrap_or(false)
}

/// The memory `surface`'s bytes are in, as a diagnostic names it: its own, for one not made over
/// the guest's pages; the guest's pages, where its base address lies in them
/// ([`base_in_pages`]); or a mapping of its own of the guest's pages, as of several pages, or of
/// one run at an offset on macOS, which IOSurface makes, and of several ranges on Linux, the
/// stand-in's view of them.
pub(crate) fn memory_kind(surface: &OsSurface) -> &'static str {
    if !over_guest_pages(surface) {
        "its own memory"
    } else if base_in_pages(surface) {
        "the guest's pages"
    } else {
        "a mapping of its own of the guest's pages"
    }
}

/// Writes `bytes` at `offset` in the bytes of `surface`, counted from its first byte, into the
/// guest's pages it was made over, where those bytes lie (from its `kIOSurfaceOffset` in the
/// pages, in their order): what the CPU writes through Metal's texture on such a surface
/// reaches (`replaceRegion:`, measured: `tests/client_direction.rs`). False, writing nothing,
/// for a surface not made over the guest's pages, and for bytes past them.
pub(crate) fn write_surface_bytes(surface: &OsSurface, offset: u64, bytes: &[u8]) -> bool {
    let Some((pages, start)) = recorded(surface, |made| (made.pages.clone(), made.offset)) else {
        return false;
    };
    let first = start.saturating_add(offset);
    let held = in_runs(&pages, &(first..first.saturating_add(bytes.len().to_u64())));
    let total: u64 = held.iter().map(|r| r.end - r.start).sum();
    if total != bytes.len().to_u64() {
        return false;
    }
    let mut from = bytes;
    for run in held {
        let (now, rest) = from.split_at((run.end - run.start).to_usize());
        // SAFETY: `run` lies in the guest's pages the surface was made over, which the host maps
        // readable and writable while the surface lives, as core holds them; `now` is the
        // caller's, apart from them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                now.as_ptr(),
                std::ptr::with_exposed_provenance_mut::<u8>(run.start.to_usize()),
                now.len(),
            );
        }
        from = rest;
    }
    true
}

/// What the OS's record of `surface` holds, read by `read`; `None` for a surface it did not
/// record.
fn recorded<T>(surface: &OsSurface, read: impl FnOnce(&MadeOver) -> T) -> Option<T> {
    host_os::recorded_mut(surface, |made| read(made))
}

/// What a surface was made over: the guest's pages, in its bytes' order, merged into runs, and
/// its bytes' offset into its allocation (`kIOSurfaceOffset`); whether the GPU's writes through
/// it never reach them ([`gpu_read_only`]), the alias of its allocation its buffers lie over
/// while one lives, where they lie over one ([`surface_alias`]), and the bytes a texture over it
/// last named as the guest's pages ([`named_bytes`]). Its pages and offset are those it was made
/// with, which never change.
#[derive(Debug, Clone, Default)]
pub(crate) struct MadeOver {
    pub(crate) pages: Vec<std::ops::Range<u64>>,
    pub(crate) offset: u64,
    pub(crate) gpu_read_only: bool,
    pub(crate) alias: Weak<Alias>,
    pub(crate) named: Option<NamedBytes>,
}

/// A surface's bytes `bytes`, from its base address, as [`named_bytes`] named them: their host
/// memory ([`surface_bytes_memory`]) named as the guest's pages are ([`crate::remap::as_guests`]),
/// at the epoch `epoch` of core's mappings of them.
#[derive(Debug, Clone)]
pub(crate) struct NamedBytes {
    bytes: std::ops::Range<u64>,
    epoch: u64,
    memory: Arc<[std::ops::Range<u64>]>,
}

/// The host memory of `surface`'s bytes `bytes` (a first byte and an end, from its base address,
/// [`surface_bytes_memory`]) named as the guest's pages are ([`crate::remap::as_guests`]), as a
/// texture made over the surface is checked against the textures heaps keep apart
/// ([`crate::gpu::over_pages`]). A display's swap makes a texture over the surface it presents
/// each frame, over the same bytes, and an x86-64 guest's 1080p surface has 2,026 ranges, so the
/// surface's record keeps what was named last: the same bytes at the same epoch of core's
/// mappings ([`crate::remap::epoch`]) name the same memory, as the surface's pages and offset
/// never change, and are given as they were named then; other bytes, a later epoch or a surface
/// the OS did not record are named again. The epoch is read before the record, so what is given
/// is what naming would answer at that read.
pub(crate) fn named_bytes(
    surface: &OsSurface,
    bytes: std::ops::Range<u64>,
) -> Arc<[std::ops::Range<u64>]> {
    let now = crate::remap::epoch();
    let kept = recorded(surface, |made| {
        (made.named.as_ref())
            .filter(|named| named.bytes == bytes && named.epoch == now)
            .map(|named| Arc::clone(&named.memory))
    });
    if let Some(memory) = kept.flatten() {
        return memory;
    }
    let memory = surface_bytes_memory(surface, std::slice::from_ref(&bytes));
    let (memory, epoch) = crate::remap::as_guests_at(memory);
    let memory: Arc<[std::ops::Range<u64>]> = memory.into();
    let named = NamedBytes {
        bytes,
        epoch,
        memory: Arc::clone(&memory),
    };
    host_os::recorded_mut(surface, |made| made.named = Some(named));
    memory
}

/// The runs of `pages`, each a host first byte and a length, those that follow on in the host's
/// memory merged, each a first byte and an end.
pub(crate) fn page_runs(pages: impl IntoIterator<Item = (u64, u64)>) -> Vec<std::ops::Range<u64>> {
    let mut runs: Vec<std::ops::Range<u64>> = Vec::new();
    for (first, length) in pages {
        let end = first.saturating_add(length);
        match runs.last_mut() {
            Some(run) if run.end == first => run.end = end,
            _ => runs.push(first..end),
        }
    }
    runs
}

#[cfg(test)]
pub(crate) mod tests {
    use reims_vgpu_core::host::{HostPtr, PhysicalMemoryRange};
    use reims_vgpu_core::platform::Handle as _;
    use reims_vgpu_core::platform::os::surface::{IoSurfaces as _, SurfaceSpec};
    use reims_vgpu_core::platform::os::vm::HostVm as _;

    use super::*;
    use crate::os::{OsSurfaces, OsVm, allocate_aliasable};

    /// The bytes of memory laid out over runs are found in each run they reach, in order
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn in_runs_finds_bytes_in_each_run_they_reach() {
        let runs = [0x10_000..0x14_000, 0x40_000..0x48_000, 0x20_000..0x24_000];
        assert_eq!(in_runs(&runs, &(0x100..0x200)), [0x10_100..0x10_200]);
        assert_eq!(
            in_runs(&runs, &(0x3f00..0x4100)),
            [0x13_f00..0x14_000, 0x40_000..0x40_100]
        );
        assert_eq!(
            in_runs(&runs, &(0x3000..0xd000)),
            [0x13_000..0x14_000, 0x40_000..0x48_000, 0x20_000..0x21_000]
        );
        assert_eq!(in_runs(&runs, &(0x10_000..0x11_000)), []);
        assert_eq!(in_runs(&[], &(0..0x100)), []);
    }

    /// A 'BGRA' surface of `width` × `height`, 4 bytes an element, rows of `width × 4`, made over
    /// fresh pages of the host's (`kIOSurfaceClientAddressRanges`) `offset` bytes into them
    /// (`kIOSurfaceOffset`), and those pages: their first byte and length, which the caller frees
    /// once the surface is gone.
    pub(crate) fn bgra_surface_over_at(
        width: u64,
        height: u64,
        offset: u64,
    ) -> (OsSurface, HostPtr, u64) {
        let page = OsVm::new().page_size();
        let length = (width * height * 4 + offset).next_multiple_of(page);
        let pages = allocate_aliasable(length).unwrap();
        let first = pages.addr().to_u64();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            offset,
            alloc_size: length,
            width,
            height,
            bytes_per_element: 4,
            bytes_per_row: width * 4,
            element_width: 1,
            element_height: 1,
            client_address_ranges: Some(
                (0..length / page)
                    .map(|i| PhysicalMemoryRange {
                        physical_address: first + i * page,
                        physical_length: page,
                    })
                    .collect(),
            ),
            ..SurfaceSpec::default()
        };
        (OsSurfaces::new().create(&spec).unwrap(), pages, length)
    }

    /// A host-only 'BGRA' surface of `width` × `height`, 4 bytes an element, rows of `width × 4`.
    pub(crate) fn bgra_surface(width: u64, height: u64) -> OsSurface {
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: width * height * 4,
            width,
            height,
            bytes_per_element: 4,
            bytes_per_row: width * 4,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        };
        OsSurfaces::new().create(&spec).unwrap()
    }

    /// The pages of [`bgra_surface_over_at`], freed.
    pub(crate) fn free(pages: HostPtr, length: u64) {
        assert!(OsVm::new().deallocate(pages, length).is_success());
    }

    /// A surface's bytes named as the guest's pages are, as its record keeps them
    /// ([`named_bytes`]): the same bytes at the same epoch of core's mappings give what was named,
    /// other bytes, or the same once core's mappings of a page under them changed, are named
    /// again, and each answer is what naming them then answers
    #[test]
    fn named_bytes_are_kept_until_the_mappings_change() {
        use reims_vgpu_core::platform::os::surface::Surface as _;
        let (s, pages, length) = bgra_surface_over_runs(4);
        let fresh = |bytes: &std::ops::Range<u64>| {
            crate::remap::as_guests(surface_bytes_memory(&s, std::slice::from_ref(bytes)))
        };
        let all = 0..s.alloc_size();
        let first = named_bytes(&s, all.clone());
        assert_eq!(&*first, &fresh(&all)[..]);
        assert!(Arc::ptr_eq(&first, &named_bytes(&s, all.clone())), "kept");
        let half = 0..s.alloc_size() / 2;
        let other = named_bytes(&s, half.clone());
        assert_eq!(&*other, &fresh(&half)[..]);
        assert!(!Arc::ptr_eq(&other, &first), "other bytes, named again");
        assert!(Arc::ptr_eq(&other, &named_bytes(&s, half.clone())));
        // The surface's second run of pages, which `half` holds, shown by a mapping of core's.
        let page = OsVm::new().page_size();
        let second = pages.addr().to_u64() + 2 * page;
        let shown = 0x55_0000_0000;
        crate::remap::guest_mapping_for_tests(second, second + page, Some(shown));
        let renamed = named_bytes(&s, half.clone());
        assert!(
            !Arc::ptr_eq(&renamed, &other),
            "named again once the mappings changed"
        );
        assert_eq!(&*renamed, &fresh(&half)[..]);
        assert!(renamed.contains(&(shown..shown + page)), "{renamed:x?}");
        crate::remap::guest_mapping_for_tests(second, second + page, None);
        let back = named_bytes(&s, half.clone());
        assert_eq!(&*back, &*other);
        assert_eq!(&*back, &fresh(&half)[..]);
        drop(s);
        free(pages, length);
    }

    /// A 'BGRA' surface of `runs` rows, each a page of the host's long, made over every other
    /// page of a fresh range (`kIOSurfaceClientAddressRanges`), so that it was made over `runs`
    /// runs of pages, as an x86 guest's 1920 × 1080 surface is over 2,025 pages on a host of
    /// 4 KiB pages; and that range, which the caller frees once the surface is gone.
    pub(crate) fn bgra_surface_over_runs(runs: u64) -> (OsSurface, HostPtr, u64) {
        let page = OsVm::new().page_size();
        let length = 2 * runs * page;
        let pages = allocate_aliasable(length).unwrap();
        let first = pages.addr().to_u64();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: runs * page,
            width: page / 4,
            height: runs,
            bytes_per_element: 4,
            bytes_per_row: page,
            element_width: 1,
            element_height: 1,
            client_address_ranges: Some(
                (0..runs)
                    .map(|i| PhysicalMemoryRange {
                        physical_address: first + 2 * i * page,
                        physical_length: page,
                    })
                    .collect(),
            ),
            ..SurfaceSpec::default()
        };
        (OsSurfaces::new().create(&spec).unwrap(), pages, length)
    }

    /// The mean time of a call of `f` over each of five rounds of `calls`, in microseconds,
    /// after one to warm up.
    pub(crate) fn microseconds_a_call(calls: u32, mut f: impl FnMut()) -> Vec<f64> {
        use reims_vgpu_core::platform::os::Clock as _;
        let clock = crate::os::OsClock::new();
        (0..6)
            .map(|_| {
                let start = clock.now_mono();
                for _ in 0..calls {
                    f();
                }
                let took = clock.now_mono().saturating_duration_since(start);
                took.as_secs_f64() * 1e6 / f64::from(calls)
            })
            .skip(1)
            .collect()
    }

    /// Prints how long `surface_bytes_memory` takes for all of a surface made over 2,025 runs
    /// of pages ([`bgra_surface_over_runs`]), as a texture on an x86 guest's 1920 × 1080 surface
    /// has it made
    #[test]
    #[ignore = "a measurement, run by hand: prints the time a call"]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn surface_bytes_memory_over_2025_runs() {
        use reims_vgpu_core::platform::os::surface::Surface as _;

        let (s, pages, length) = bgra_surface_over_runs(2025);
        let bytes = [0..s.alloc_size()];
        assert_eq!(surface_bytes_memory(&s, &bytes).len(), 1 + 2025);
        let rounds = microseconds_a_call(200, || {
            std::hint::black_box(surface_bytes_memory(&s, std::hint::black_box(&bytes)));
        });
        println!("surface_bytes_memory over 2,025 runs, µs a call: {rounds:.1?}");
        drop(s);
        free(pages, length);
    }

    /// A lookup of a surface, as an isolated task's surface is (`lookup_by_port`), a reference
    /// of its own, answers what the surface was made over: its offset, memory and GPU writes
    #[test]
    fn a_lookup_of_a_surface_answers_its_record() {
        let (s, pages, length) = bgra_surface_over_at(16, 16, 0x100);
        let t = s.lookup_by_port().unwrap();
        assert_ne!(t.id(), s.id());
        assert_eq!(surface_offset(&t), 0x100);
        assert_eq!(surface_memory(&t), surface_memory(&s));
        let page = OsVm::new().page_size();
        let guest = allocate_aliasable(page).unwrap();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: page,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_direction: Some(READ_ONLY_DIRECTION),
            client_address_ranges: Some(vec![PhysicalMemoryRange {
                physical_address: guest.addr().to_u64(),
                physical_length: page,
            }]),
            ..SurfaceSpec::default()
        };
        let read_only = OsSurfaces::new().create(&spec).unwrap();
        let looked_up = read_only.lookup_by_port().unwrap();
        assert!(gpu_read_only(&read_only) && gpu_read_only(&looked_up));
        assert!(!gpu_read_only(&s) && !gpu_read_only(&t));
        drop((t, s, looked_up, read_only));
        free(pages, length);
        free(guest, page);
    }
}
