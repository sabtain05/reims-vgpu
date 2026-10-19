//! The host pages runtime imports hold, the unmaps that wait for them ([`super::UnmapHold`]), and
//! those left to the unmap thread: the bookkeeping alone, which makes no call (`super`'s docs,
//! "Unmaps an import waits for").

use std::collections::BTreeMap;
use std::ops::Range;

/// The live holds and the unmaps waiting for them, each a range of whole pages. A leaf lock
/// guards it: nothing is called under it.
#[derive(Debug)]
pub(super) struct Holds {
    /// The next hold's number.
    next: u64,
    /// Each live hold's pages, by their first byte and the hold's number, to their end.
    held: BTreeMap<(u64, u64), u64>,
    /// The most bytes any hold took, which bounds how far below a range a hold that meets it
    /// starts.
    longest: u64,
    /// The unmaps that wait, no two sharing a page: each made whole once no hold meets it.
    pub(super) waiting: Vec<Range<u64>>,
    /// The unmaps no hold meets any more that a hold's release left to the unmap thread, until
    /// it takes them; none shares a page with another, or with one that waits.
    queued: Vec<Range<u64>>,
    /// Whether the unmap thread is making the unmaps it took last.
    making: bool,
}

impl Holds {
    /// None.
    pub(super) const fn new() -> Holds {
        Holds {
            next: 0,
            held: BTreeMap::new(),
            longest: 0,
            waiting: Vec::new(),
            queued: Vec::new(),
            making: false,
        }
    }

    /// A hold on `pages`: its number.
    pub(super) fn hold(&mut self, pages: Range<u64>) -> u64 {
        let id = self.next;
        self.next = self.next.wrapping_add(1);
        self.longest = self.longest.max(pages.end.saturating_sub(pages.start));
        self.held.insert((pages.start, id), pages.end);
        id
    }

    /// Whether a live hold meets `pages`.
    fn meets(&self, pages: &Range<u64>) -> bool {
        let from = pages.start.saturating_sub(self.longest);
        (self.held.range((from, 0)..(pages.end, 0)))
            .any(|(&(first, _), &end)| first < pages.end && pages.start < end)
    }

    /// `pages` show other pages from now on (an overwrite, or an unmap made now): what waited
    /// to be unmapped there, or was left to the unmap thread, no longer is.
    pub(super) fn replaced(&mut self, pages: &Range<u64>) {
        if pages.is_empty() {
            return;
        }
        cut(&mut self.waiting, pages);
        cut(&mut self.queued, pages);
    }

    /// An unmap of `pages`: `true` where a live hold meets them, and the unmap waits, whole, for
    /// the last hold that meets it; `false` where the caller makes it now.
    pub(super) fn unmap(&mut self, pages: Range<u64>) -> bool {
        self.replaced(&pages);
        if pages.is_empty() || !self.meets(&pages) {
            return false;
        }
        self.waiting.push(pages);
        true
    }

    /// Whether an unmap of a page of `pages` waits.
    #[cfg(test)]
    pub(super) fn waits(&self, pages: &Range<u64>) -> bool {
        (self.waiting.iter()).any(|waiting| waiting.start < pages.end && pages.start < waiting.end)
    }

    /// The hold `id` on the pages from `first` goes: the unmaps no live hold meets any more,
    /// which the caller makes.
    pub(super) fn release(&mut self, first: u64, id: u64) -> Vec<Range<u64>> {
        self.held.remove(&(first, id));
        if self.waiting.is_empty() {
            return Vec::new();
        }
        let (free, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.waiting)
            .into_iter()
            .partition(|waiting| !self.meets(waiting));
        self.waiting = kept;
        free
    }

    /// Leaves `free`, unmaps [`Holds::release`] answered, to the unmap thread.
    pub(super) fn queue(&mut self, free: Vec<Range<u64>>) {
        self.queued.extend(free);
    }

    /// Whether an unmap waits for the unmap thread to take it.
    pub(super) fn queued(&self) -> bool {
        !self.queued.is_empty()
    }

    /// The unmap thread takes every unmap left to it, which it makes under no lock until
    /// [`Holds::made`].
    pub(super) fn take_queued(&mut self) -> Vec<Range<u64>> {
        let taken = std::mem::take(&mut self.queued);
        self.making |= !taken.is_empty();
        taken
    }

    /// The unmap thread made what it took last.
    pub(super) fn made(&mut self) {
        self.making = false;
    }

    /// Whether the unmap thread has an unmap left to make, taken or not.
    #[cfg(test)]
    pub(super) fn unmapping(&self) -> bool {
        self.making || !self.queued.is_empty()
    }
}

/// `ranges` less what of them lies in `pages`.
fn cut(ranges: &mut Vec<Range<u64>>, pages: &Range<u64>) {
    if ranges.is_empty() {
        return;
    }
    let mut kept = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if range.end <= pages.start || pages.end <= range.start {
            kept.push(range);
            continue;
        }
        if range.start < pages.start {
            kept.push(range.start..pages.start);
        }
        if pages.end < range.end {
            kept.push(pages.end..range.end);
        }
    }
    *ranges = kept;
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 0x4000;

    /// `n` pages from page `at`.
    fn pages(at: u64, n: u64) -> Range<u64> {
        at * PAGE..(at + n) * PAGE
    }

    /// An unmap no hold meets is made now; one a hold meets waits, whole, until the last hold
    /// that meets it goes, and is made once
    #[test]
    fn an_unmap_waits_for_every_hold_that_meets_it() {
        let mut h = Holds::new();
        assert!(!h.unmap(pages(10, 2)), "no hold");
        let a = h.hold(pages(10, 2));
        let b = h.hold(pages(11, 4));
        assert!(!h.unmap(pages(20, 1)), "pages no hold meets");
        assert!(
            h.unmap(pages(8, 4)),
            "pages the first hold meets, with others"
        );
        assert_eq!(
            h.release(10 * PAGE, a),
            Vec::<Range<u64>>::new(),
            "b meets it"
        );
        assert_eq!(h.release(11 * PAGE, b), vec![pages(8, 4)]);
        assert_eq!(
            h.release(11 * PAGE, b),
            Vec::<Range<u64>>::new(),
            "made once"
        );
    }

    /// Two holds on the same pages each keep an unmap of them waiting
    #[test]
    fn holds_on_the_same_pages_count_apart() {
        let mut h = Holds::new();
        let a = h.hold(pages(4, 1));
        let b = h.hold(pages(4, 1));
        assert!(h.unmap(pages(4, 1)));
        assert!(h.release(4 * PAGE, b).is_empty());
        assert_eq!(h.release(4 * PAGE, a), vec![pages(4, 1)]);
    }

    /// A long hold far below a range still meets it
    #[test]
    fn a_long_hold_meets_pages_far_into_it() {
        let mut h = Holds::new();
        let long = h.hold(pages(0, 1000));
        let _short = h.hold(pages(2000, 1));
        assert!(h.unmap(pages(999, 1)));
        assert!(!h.unmap(pages(1000, 1)), "past its end");
        assert_eq!(h.release(0, long), vec![pages(999, 1)]);
    }

    /// Unmaps left to the unmap thread wait until it takes them, all at once; pages mapped anew,
    /// or unmapped now, under one it has not taken leave it, as they leave a waiting unmap
    #[test]
    fn the_unmap_thread_takes_what_was_left_to_it() {
        let mut h = Holds::new();
        let a = h.hold(pages(0, 8));
        assert!(h.unmap(pages(0, 8)));
        let free = h.release(0, a);
        assert_eq!(free, vec![pages(0, 8)]);
        h.queue(free);
        assert!(h.queued() && h.unmapping());
        h.replaced(&pages(2, 2));
        assert!(
            !h.unmap(pages(6, 1)),
            "no hold meets it: made now, and out of the queue"
        );
        assert_eq!(h.take_queued(), vec![pages(0, 2), pages(4, 2), pages(7, 1)]);
        assert!(!h.queued() && h.unmapping(), "taken, not yet made");
        h.made();
        assert!(!h.unmapping());
    }

    /// Pages mapped anew, or unmapped now, under a waiting unmap leave it: what of it they cover
    /// is never unmapped later, and the rest still waits
    #[test]
    fn pages_replaced_under_a_waiting_unmap_leave_it() {
        let mut h = Holds::new();
        let a = h.hold(pages(0, 8));
        assert!(h.unmap(pages(0, 8)));
        h.replaced(&pages(2, 2));
        assert!(h.unmap(pages(6, 1)), "waits again, alone");
        assert_eq!(
            h.release(0, a),
            vec![pages(0, 2), pages(4, 2), pages(7, 1), pages(6, 1)]
        );
    }
}
