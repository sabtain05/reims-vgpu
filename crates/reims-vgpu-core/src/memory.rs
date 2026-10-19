//! Guest memory as the host hands it over: the memory map and the mappings made from it, the
//! page-table walk of a task, the mapped-range tracker and the cursor over host ranges.
//!
//! - [`descriptor`]: the host's ranges, in the order the host added them.
//! - [`map`]: [`MemoryMap`](map::MemoryMap), the guest's physical ranges sorted at init, and the
//!   mappings it makes (`mach_vm_remap` of each piece into a fresh or given region).
//! - [`mapping`]: [`Mapping`](mapping::Mapping), a host range over guest memory, and its four
//!   teardowns.
//! - [`page_table`]: [`PageTableWalker`](page_table::PageTableWalker), a task's 1-3 level page
//!   table walked at use time.
//! - [`tracker`]: [`RangeTracker`](tracker::RangeTracker), which task pages the host has mapped,
//!   as runs in up to 128 blocks of 256.
//! - [`cursor`]: [`Cursor`](cursor::Cursor), a read (and write) position over a list of host
//!   ranges.
//!
//! # Edges the guest and host see
//!
//! Each of these is pinned by a test, and the code notes it where it happens.
//!
//! - **Map validation order** (`map`): the descriptor's ranges are summed in the host's order
//!   (an overflow refuses the map before anything is sorted), sorted by start and then length
//!   (zero-length ranges before the range with a length that shares their start), then
//!   checked against the previous range's end only. An isolated task's service copies the
//!   host's map, neither sorted nor validated again.
//! - **Lookups** (`map`): the first containing range decides and a straddle is refused, while a
//!   range mapping splits across contiguous ranges. Guest physical 0 is refused even with
//!   length 0. A write tests its source, not the translation. A failed protect deallocates and
//!   still returns the mapping, whose release frees nothing more; a length of 0 passes the page
//!   assert.
//! - **Mappings** (`mapping`): the last release runs the teardown, and a mapping of the root
//!   task or of a host task unmaps through its device while the device's fields live, so
//!   mappings released during the device's teardown still unmap; one released after skips it.
//! - **The walk** (`page_table`): its refusals in order, before the zero-root assert; the read
//!   block's and the visitor's answers ignored; level indices truncated to 32 bits and the root
//!   read limited to the touched entries; a hole raises after the visits before it.
//! - **The tracker** (`tracker`): the run-block structure, adds stopping at 240 runs a block
//!   and `remove` growing one to 256, the last-probe binary search, the emptied block that
//!   stays with a `{0,0}` hull, and the raises after partial work.
//! - **The cursor** (`cursor`): the leading empty range of a pointer cursor, advancing that
//!   saturates without walking, and addresses summed as integers, so a NULL range faults at
//!   the copy.
//!
//! # Outcomes
//!
//! Raises are [`RaiseKind`]s on the raise channel, deaths [`DeathKind`]s.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

pub mod cursor;
pub mod descriptor;
pub mod map;
pub mod mapping;
pub mod page_table;
pub mod tracker;

use std::fmt;

use crate::raise::death::{self, DeathSite};
use crate::raise::{DeathId, OnFault, Raise, RaiseId};

/// What memory raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// A cursor read past its length ("record not readable: …").
    CursorReadFailed,
    /// A walk from a start that is not page-aligned.
    WalkMisalignedStart,
    /// A walk over a length that is not whole pages.
    WalkMisalignedLength,
    /// A walk over nothing.
    WalkZeroLength,
    /// A walk whose last byte's address overflows.
    WalkEndOverflow,
    /// The root table's read refused its indices.
    WalkRootUnread,
    /// A zero entry in a walk that allows no holes.
    WalkHole,
    /// A run subtracted from inside another.
    RunSubtractInside,
    /// A page-aligned run over a length of 0 or not whole pages.
    TrackerAlignedLength,
    /// A page-aligned run from an offset that is not page-aligned.
    TrackerAlignedOffset,
    /// A page run over a length of 0.
    TrackerZeroLength,
    /// An add needing a 129th block after one compaction or split.
    TrackerTooComplex,
    /// More than 16 unmapped runs, found while scanning a block.
    TrackerUnmappedTooComplex,
    /// More than 16 unmapped runs, found after the last block.
    TrackerUnmappedTooComplexTail,
}

impl RaiseKind {
    /// The dotted id.
    pub const fn name(self) -> &'static str {
        match self {
            RaiseKind::CursorReadFailed => "memory.cursor.read_failed",
            RaiseKind::WalkMisalignedStart => "memory.walk.misaligned_start",
            RaiseKind::WalkMisalignedLength => "memory.walk.misaligned_length",
            RaiseKind::WalkZeroLength => "memory.walk.zero_length",
            RaiseKind::WalkEndOverflow => "memory.walk.end_overflow",
            RaiseKind::WalkRootUnread => "memory.walk.root_unread",
            RaiseKind::WalkHole => "memory.walk.hole",
            RaiseKind::RunSubtractInside => "memory.tracker.subtract_inside",
            RaiseKind::TrackerAlignedLength => "memory.tracker.aligned_length",
            RaiseKind::TrackerAlignedOffset => "memory.tracker.aligned_offset",
            RaiseKind::TrackerZeroLength => "memory.tracker.zero_length",
            RaiseKind::TrackerTooComplex => "memory.tracker.too_complex",
            RaiseKind::TrackerUnmappedTooComplex => "memory.tracker.unmapped_too_complex",
            RaiseKind::TrackerUnmappedTooComplexTail => "memory.tracker.unmapped_too_complex_tail",
        }
    }

    /// Raises this with `reason` (its text is diagnostics), in a walk reached both ways: with
    /// [`OnFault::Die`] it dies here, with [`OnFault::Return`] it is the raise to hand on.
    #[track_caller]
    pub(crate) fn raise(self, on: OnFault, reason: &str) -> Raise {
        let raised = on.throw::<std::convert::Infallible>(Raise::new(self).with_detail(reason));
        match raised {
            Err(r) => r,
            Ok(never) => match never {},
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::Memory(k)
    }
}

/// Where memory ends the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A cursor step past its list of ranges (unreachable: the lengths keep it inside).
    CursorRangeIndex,
    /// A cursor copy past the caller's buffer (unreachable: the count keeps it inside).
    CursorBufferEnd,
    /// A cursor read through a NULL or near-NULL range.
    CursorReadNull,
    /// A cursor write through a NULL or near-NULL range.
    CursorWriteNull,
    /// A map read whose translation faults.
    MapReadFault,
    /// A map write through the NULL translation of an address outside the map.
    MapWriteNull,
    /// A range mapping of a length that is not whole pages.
    MappingLengthUnaligned,
    /// A root-task mapping at NULL.
    RootMappingNullAddress,
    /// A root-task mapping of length 0.
    RootMappingZeroLength,
    /// A root-task mapping whose device is gone.
    RootMappingNoDevice,
    /// A walk reading the table at guest page 0.
    WalkZeroTable,
    /// A table read past the table's entries (4096, 1024 on x86-64).
    WalkReadPastTable,
    /// A walk of depth 0.
    WalkNoLevels,
    /// A walk reading an entry past its table (a stale index, or a truncated one).
    WalkEntryPastTable,
    /// Two runs in no order `compare` knows.
    RunCompareUnordered,
    /// An add to a block of 256 runs.
    BlockAddFull,
    /// A block's add meeting a run it cannot follow.
    BlockAddUnordered,
    /// A remove from a block of 256 runs.
    BlockRemoveFull,
    /// A block's remove meeting an equal run after one it marked.
    BlockRemoveEqualAfterMark,
    /// A block's removal of runs past its count.
    BlockRemovePastCount,
    /// An insert into a block of 256 runs.
    BlockInsertFull,
    /// A split whose halves do not add up (an odd count).
    BlockSplitCount,
    /// A split whose halves overlap.
    BlockSplitOverlap,
    /// The tracker reading a block slot that holds none.
    TrackerSlotEmpty,
    /// The add's target block moved.
    TrackerTargetMoved,
    /// The tracker's add meeting a block it cannot follow.
    TrackerAddUnordered,
    /// The tracker removing a block from an empty slot.
    TrackerRemoveEmptySlot,
    /// The tracker removing a block with none left.
    TrackerRemoveNoBlocks,
    /// A compaction past the last block slot.
    TrackerCompactPastBlocks,
    /// A compaction stealing from an empty block.
    TrackerCompactEmptyNext,
    /// A test that covered the range yet left some of it.
    TrackerTestLeftover,
    /// An unmapped run emptied by a run inside it.
    TrackerVisitEmptyRun,
}

impl DeathKind {
    /// The dotted id.
    pub const fn name(self) -> &'static str {
        match self {
            DeathKind::CursorRangeIndex => "memory.cursor.range_index",
            DeathKind::CursorBufferEnd => "memory.cursor.buffer_end",
            DeathKind::CursorReadNull => "memory.cursor.read_null",
            DeathKind::CursorWriteNull => "memory.cursor.write_null",
            DeathKind::MapReadFault => "memory.map.read_fault",
            DeathKind::MapWriteNull => "memory.map.write_null",
            DeathKind::MappingLengthUnaligned => "memory.map.mapping_length_unaligned",
            DeathKind::RootMappingNullAddress => "memory.mapping.root_null_address",
            DeathKind::RootMappingZeroLength => "memory.mapping.root_zero_length",
            DeathKind::RootMappingNoDevice => "memory.mapping.root_no_device",
            DeathKind::WalkZeroTable => "memory.walk.zero_table",
            DeathKind::WalkReadPastTable => "memory.walk.read_past_table",
            DeathKind::WalkNoLevels => "memory.walk.no_levels",
            DeathKind::WalkEntryPastTable => "memory.walk.entry_past_table",
            DeathKind::RunCompareUnordered => "memory.tracker.compare_unordered",
            DeathKind::BlockAddFull => "memory.tracker.block_add_full",
            DeathKind::BlockAddUnordered => "memory.tracker.block_add_unordered",
            DeathKind::BlockRemoveFull => "memory.tracker.block_remove_full",
            DeathKind::BlockRemoveEqualAfterMark => "memory.tracker.block_remove_equal_after_mark",
            DeathKind::BlockRemovePastCount => "memory.tracker.block_remove_past_count",
            DeathKind::BlockInsertFull => "memory.tracker.block_insert_full",
            DeathKind::BlockSplitCount => "memory.tracker.block_split_count",
            DeathKind::BlockSplitOverlap => "memory.tracker.block_split_overlap",
            DeathKind::TrackerSlotEmpty => "memory.tracker.slot_empty",
            DeathKind::TrackerTargetMoved => "memory.tracker.target_moved",
            DeathKind::TrackerAddUnordered => "memory.tracker.add_unordered",
            DeathKind::TrackerRemoveEmptySlot => "memory.tracker.remove_empty_slot",
            DeathKind::TrackerRemoveNoBlocks => "memory.tracker.remove_no_blocks",
            DeathKind::TrackerCompactPastBlocks => "memory.tracker.compact_past_blocks",
            DeathKind::TrackerCompactEmptyNext => "memory.tracker.compact_empty_next",
            DeathKind::TrackerTestLeftover => "memory.tracker.test_leftover",
            DeathKind::TrackerVisitEmptyRun => "memory.tracker.visit_empty_run",
        }
    }

    /// The kind of death there (diagnostics: every death aborts).
    const fn fault(self) -> death::DeathKind {
        match self {
            DeathKind::CursorRangeIndex
            | DeathKind::CursorBufferEnd
            | DeathKind::WalkReadPastTable
            | DeathKind::WalkNoLevels
            | DeathKind::WalkEntryPastTable => death::DeathKind::OutOfBounds,
            DeathKind::CursorReadNull
            | DeathKind::CursorWriteNull
            | DeathKind::MapReadFault
            | DeathKind::MapWriteNull
            | DeathKind::TrackerSlotEmpty => death::DeathKind::NullDeref,
            DeathKind::BlockAddUnordered | DeathKind::TrackerAddUnordered => {
                death::DeathKind::Abort
            }
            DeathKind::MappingLengthUnaligned
            | DeathKind::RootMappingNullAddress
            | DeathKind::RootMappingZeroLength
            | DeathKind::RootMappingNoDevice
            | DeathKind::WalkZeroTable
            | DeathKind::RunCompareUnordered
            | DeathKind::BlockAddFull
            | DeathKind::BlockRemoveFull
            | DeathKind::BlockRemoveEqualAfterMark
            | DeathKind::BlockRemovePastCount
            | DeathKind::BlockInsertFull
            | DeathKind::BlockSplitCount
            | DeathKind::BlockSplitOverlap
            | DeathKind::TrackerTargetMoved
            | DeathKind::TrackerRemoveEmptySlot
            | DeathKind::TrackerRemoveNoBlocks
            | DeathKind::TrackerCompactPastBlocks
            | DeathKind::TrackerCompactEmptyNext
            | DeathKind::TrackerTestLeftover
            | DeathKind::TrackerVisitEmptyRun => death::DeathKind::Assert,
        }
    }

    /// The site a death here records (`testkit::capture`) and names on stderr: the dotted id, at
    /// line 0. Also what the guest accessors take for a fault at a copy.
    pub const fn site(self) -> DeathSite {
        DeathSite {
            kind: self.fault(),
            file: self.name(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> DeathId {
        DeathId::Memory(k)
    }
}
