//! The budget of kernel mappings (VMAs) the library's views of memory may hold.
//!
//! A view of scattered guest pages is a mapping per run of them ([`super::vm`]), and the kernel
//! refuses any map past `vm.max_map_count` mappings in the process: past that, not only views
//! fail but every map the process makes, QEMU's, the C library's and the GPU driver's among them.
//! So the views stop short of the limit: a view, a range a view can be made of, an import's
//! writable twin or an IOSurface stand-in's memory that would take the mappings the library holds
//! past `vm.max_map_count - RESERVE` ends the process before anything is mapped, naming what it
//! would have mapped, the mappings and the limit ([`admit`]), and the mappings the rest of the
//! process makes are never refused for the views' sake. macOS has no such limit, so Metal's call
//! (a replace, a heap or a buffer over the guest's pages, an IOSurface) goes on where the views
//! run out: no answer of Metal's fits, and the `KERN_NO_SPACE`, nil or NO the call would show
//! otherwise would be one Metal never gives there.
//!
//! The mappings the library holds are counted exactly where it makes them: the backing table's
//! runs, each protect or unmap that cuts one included ([`super::files::mappings`]), the import
//! twins' pieces ([`super::import`]) and the stand-ins' memory ([`super::vm::OwnMemory`]). What it
//! does not count is the reserve's to cover ([`RESERVE`]).
//!
//! The limit is read when the first view is weighed, and again when one would pass it, as it may
//! have been raised since. A guest whose RAM in 16 KiB pieces would pass the budget is warned of
//! when its memory is leased ([`warn_if_short`]), with the setting that raises it.

use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};

use super::files;

/// The mappings left to the rest of the process: QEMU's own, the C library's, the GPU driver's
/// and the runtime's (device memory maps), threads' stacks, and the library's anonymous
/// reservations with the gaps a view leaves in one, none of which the library counts.
///
/// No rule bounds them: what QEMU, the driver and the runtime map grows with the guest's work and
/// differs by driver and release, and nothing the library can read beforehand says how far. So
/// the reserve is a margin, not a bound: a quarter of the kernel's default `vm.max_map_count`
/// (65,530), so that the views may hold three quarters of a default limit. One measurement backs
/// it, no more: on a 12 GiB x86-64 guest's boot and a Safari and Maps session on an RTX 5080
/// laptop host, the
/// process held about 1,700 mappings before the guest's GPU work began and at most 4,600 besides
/// its views of guest memory. Where the views run out of it they end the process, naming the
/// numbers ([`admit`]); where the rest of the process does, the kernel refuses its maps, and the
/// library's own refusal for want of mappings ends the process the same way
/// ([`super::vm`]'s `no_space`).
pub(super) const RESERVE: u64 = 16_384;

/// The piece guest memory comes in for the warning's estimate: 16 KiB, the run nearly every
/// view of an x86-64 guest's memory was made of when measured (and an arm64 guest's page).
const GUEST_PIECE: u64 = 0x4000;

/// `vm.max_map_count` as last read; 0 before the first read.
static LIMIT: AtomicU64 = AtomicU64::new(0);

/// The setting the warnings name.
const SETTING: &str = "vm.max_map_count";

#[cfg(test)]
thread_local! {
    /// A test's limit, in place of the kernel's, on the thread that sets it.
    static TEST_LIMIT: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Sets the calling thread's limit in place of the kernel's, or the kernel's again with `None`.
#[cfg(test)]
pub(super) fn set_test_limit(limit: Option<u64>) {
    TEST_LIMIT.with(|l| l.set(limit));
}

/// `vm.max_map_count` read now; `u64::MAX` where it does not read, leaving the kernel's own
/// refusal.
fn read_limit() -> u64 {
    std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .ok()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or(u64::MAX)
}

/// The limit as last read, reading it the first time; a test's, where the thread set one.
fn limit() -> u64 {
    #[cfg(test)]
    if let Some(limit) = TEST_LIMIT.with(std::cell::Cell::get) {
        return limit;
    }
    match LIMIT.load(Ordering::Relaxed) {
        0 => reread(),
        limit => limit,
    }
}

/// The limit read again, and kept; a test's, where the thread set one.
fn reread() -> u64 {
    #[cfg(test)]
    if let Some(limit) = TEST_LIMIT.with(std::cell::Cell::get) {
        return limit;
    }
    let limit = read_limit();
    LIMIT.store(limit, Ordering::Relaxed);
    limit
}

/// The mappings the library's views may hold under `limit`.
fn budget(limit: u64) -> u64 {
    limit.saturating_sub(RESERVE)
}

/// The import twins' mappings.
static TWIN_MAPPINGS: AtomicU64 = AtomicU64::new(0);

/// The IOSurface stand-ins' mappings of their own memory.
static OWN_MAPPINGS: AtomicU64 = AtomicU64::new(0);

/// Counts `count` more import twin mappings, or fewer.
pub(super) fn twins_hold(count: u64, holding: bool) {
    count_on(&TWIN_MAPPINGS, count, holding);
}

/// Counts `count` more stand-in mappings, or fewer.
pub(super) fn own_hold(count: u64, holding: bool) {
    count_on(&OWN_MAPPINGS, count, holding);
}

/// Adds `count` to `counter` while `holding`, else takes it away.
fn count_on(counter: &AtomicU64, count: u64, holding: bool) {
    if holding {
        counter.fetch_add(count, Ordering::Relaxed);
    } else {
        counter.fetch_sub(count, Ordering::Relaxed);
    }
}

/// The import twins' mappings now.
pub(super) fn twin_mappings() -> u64 {
    TWIN_MAPPINGS.load(Ordering::Relaxed)
}

/// The stand-ins' mappings of their own memory now.
pub(super) fn own_mappings() -> u64 {
    OWN_MAPPINGS.load(Ordering::Relaxed)
}

/// The budget under the limit as last read.
pub(super) fn budget_now() -> u64 {
    budget(limit())
}

/// The mappings the library's views hold now (module docs).
pub(super) fn held() -> u64 {
    files::mappings()
        .saturating_add(TWIN_MAPPINGS.load(Ordering::Relaxed))
        .saturating_add(OWN_MAPPINGS.load(Ordering::Relaxed))
}

/// Lets `what` add up to `adding` mappings, or ends the process where the views would pass the
/// budget with them, at the limit read again (module docs): the death names `what`, the
/// mappings the views hold and the process holds, the limit, the reserve and the setting that
/// gives room. Threads that weigh views at once may pass it by what they add between them, which
/// the reserve covers.
pub(super) fn admit(adding: u64, what: &'static str) {
    let held = held();
    let wanted = held.saturating_add(adding);
    if wanted <= budget(limit()) {
        return;
    }
    let limit = reread();
    if wanted <= budget(limit) {
        return;
    }
    let (mappings, _) = super::vm::map_census();
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: "mmap",
        },
        &exhausted(what, adding, held, limit, &mappings),
    )
}

/// The line of [`admit`]'s death: `what` would add `adding` mappings to the `held` the views hold
/// under `limit`, the process holding `mappings`.
fn exhausted(what: &str, adding: u64, held: u64, limit: u64, mappings: &str) -> String {
    let room = held
        .saturating_add(adding)
        .saturating_add(RESERVE)
        .next_power_of_two()
        .max(1 << 20);
    format!(
        "the library's views of memory ran out of kernel mappings: {what} would add up to \
         {adding} to the {held} they hold, past their budget of {}, {SETTING} {limit} less the \
         {RESERVE} kept for the rest of the process (which holds {mappings} in all); Metal's call \
         goes on where macOS has no such limit, and `sysctl -w {SETTING}={room}` or more gives \
         room",
        budget(limit),
    )
}

/// Warns once, as a guest's RAM of `guest_ram` bytes is leased, where views of all of it in
/// 16 KiB pieces would pass the budget: past that much of it, a view ends the process
/// ([`admit`]).
pub(super) fn warn_if_short(guest_ram: u64) {
    let limit = reread();
    let needed = guest_ram / GUEST_PIECE;
    if needed <= budget(limit) {
        return;
    }
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        let mib = budget(limit).saturating_mul(GUEST_PIECE) >> 20;
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "{SETTING} is {limit}: views of guest memory end the process past {} mappings, about \
             {mib} MiB of the guest's {} MiB in 16 KiB pieces; `sysctl -w {SETTING}={}` or more \
             lets them cover it",
            budget(limit),
            guest_ram >> 20,
            needed.saturating_add(RESERVE).next_power_of_two().max(1 << 20),
        );
    });
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::num::ToU64 as _;
    use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _, VmProt};

    use super::super::{LinuxVm, import, vm};
    use super::*;

    const PAGE: u64 = 0x1000;

    /// Sets the thread's limit for a test, and the kernel's again when it drops.
    struct Limit;

    impl Limit {
        /// Room for `room` more mappings than the views hold now, past the reserve.
        fn room(room: u64) -> Limit {
            set_test_limit(Some(held() + RESERVE + room));
            Limit
        }
    }

    impl Drop for Limit {
        fn drop(&mut self) {
            set_test_limit(None);
        }
    }

    /// Whether `f` ends the process at the budget's death (`mmap`).
    fn dies_at_the_budget<R>(f: impl FnOnce() -> R) -> bool {
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;
        catch_death(f).is_err_and(|site| (site.kind, site.file) == (DeathKind::Abort, "mmap"))
    }

    /// A view that would take the views past the budget ends the process before anything is
    /// mapped: its target keeps no file and reads its zeros; with room, the same view is made
    #[test]
    fn a_view_past_the_budget_dies_before_anything_is_mapped() {
        let v = LinuxVm::new();
        let src = vm::allocate_aliasable(64 * PAGE).unwrap();
        let at = v.allocate_anywhere(64 * PAGE).unwrap();
        let page = |base: HostAddr, i: u64| base.wrapping_add(i * PAGE);
        let limit = Limit::room(8);
        // Pages in reverse: a mapping each, until the budget ends the process at one.
        let died = (0..64).find(|&i| {
            dies_at_the_budget(|| v.remap_shared_overwrite(page(at, i), page(src, 63 - i), PAGE))
        });
        let i = died.expect("a view past the budget");
        let target = page(at, i).addr().to_u64();
        assert!(files::pieces(target, PAGE).is_none(), "no file mapped");
        // SAFETY: the reservation's page, mapped read and write.
        assert_eq!(unsafe { page(at, i).as_ptr::<u8>().read_volatile() }, 0);
        drop(limit);
        v.remap_shared_overwrite(page(at, i), page(src, 63 - i), PAGE)
            .unwrap();
        assert!(files::pieces(target, PAGE).is_some());
        assert!(v.deallocate(at, 64 * PAGE).is_success());
        assert!(v.deallocate(src, 64 * PAGE).is_success());
    }

    /// A range a view can be made of, an import's twin and an IOSurface stand-in's memory end
    /// the process past the budget too; with room, the twin is made
    #[test]
    fn ranges_twins_and_stand_ins_past_the_budget_die() {
        let v = LinuxVm::new();
        let at = vm::allocate_aliasable(2 * PAGE).unwrap();
        assert!(v.protect_max(at, 2 * PAGE, VmProt::READ).is_ok());
        let p = std::ptr::NonNull::new(at.as_ptr::<u8>()).unwrap();
        {
            let _limit = Limit::room(0);
            assert!(dies_at_the_budget(|| vm::allocate_aliasable(PAGE)));
            assert!(dies_at_the_budget(|| import::import_view(p, PAGE)));
            assert!(dies_at_the_budget(|| vm::OwnMemory::new(PAGE)));
        }
        assert_ne!(import::import_view(p, PAGE).at(), p, "a twin with room");
        assert!(v.deallocate(at, 2 * PAGE).is_success());
    }

    /// The death names what would be mapped, the mappings, the limit, the reserve and a setting
    /// with room
    #[test]
    fn the_death_names_the_numbers() {
        let line = exhausted("a view of guest pages", 3, 49_140, 65_530, "52000");
        for part in [
            "ran out of kernel mappings",
            "a view of guest pages would add up to 3 to the 49140 they hold",
            "past their budget of 49146, vm.max_map_count 65530 less the 16384 kept",
            "holds 52000 in all",
            "`sysctl -w vm.max_map_count=1048576`",
        ] {
            assert!(line.contains(part), "{part}: {line}");
        }
    }

    /// The budget is the limit less the reserve, and none where the limit does not read
    #[test]
    fn the_budget_is_the_limit_less_the_reserve() {
        assert_eq!(budget(65_530), 65_530 - 16_384);
        assert_eq!(budget(1_048_576), 1_048_576 - 16_384);
        assert_eq!(budget(100), 0);
        assert_eq!(budget(u64::MAX), u64::MAX - 16_384);
    }
}
