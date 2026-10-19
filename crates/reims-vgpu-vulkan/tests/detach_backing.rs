//! Detached buffers and heaps as core's isolated tasks make and use them
//! (`resources/buffer/detach.rs`, `resources/heap/detach.rs`), on the Metal platform and on this
//! crate's: a no-copy buffer over a temporary allocation and a heap over its address range,
//! detached at once, then replaced by guest pages, scattered, read-only or not, detached and
//! replaced again. Each platform's answers, the GPU address, the deallocator's runs, the bytes
//! `contents` reads, and what the GPU reads and writes in the guest's pages, are the same: under a
//! read-only replace the GPU's fill never reaches the pages, a copy later in its command buffer
//! reads the fill, or the pages where the GPU's caches no longer hold it ([`cached_or_pages`]),
//! and one in the next command buffer the pages. Measured on
//! `AppleM3Max-40c-25G229`.
//!
//! Command buffers that use a detached buffer or heap ([`detached_use`]): both platforms fail
//! the submission that uses it, run the command buffer's other submissions, decide a submission
//! behind an event wait as they submit it, and fail a use through a texture over a buffer or
//! placed in a heap released detached. A command buffer committed behind an event wait whose
//! buffer is detached, replaced, or both before the signal ([`behind_a_wait`]): this crate's
//! fails a use detached then as Metal's does, in the first submission or a later one, and dies
//! at a use replaced then, also in a submission after one a detached use failed, where Metal
//! runs it over the new pages (`src/gpu/command.rs`'s docs). Behind waits on plain events the
//! GPU signals, Metal decides the submission as it schedules the command buffer, once for each
//! event a command buffer that signals it at the value waited for or past it was committed, or
//! at its commit where the event holds such a value, and from then on the objects it uses are in
//! flight, as below ([`behind_a_gpu_wait`], [`behind_gpu_waits`], measured on Metal only). A
//! buffer placed in a heap over address ranges detaches the heap on both, and one in a plain
//! heap, as the plain heap, never detaches ([`placed_detach`]).
//!
//! A detach while a command buffer submitted and not yet completed took the object answers NO
//! on both, a heap's at once and a buffer's once the command buffer completed, where Metal's
//! buffer answers YES in some runs from the same state, an answer its test does not compare
//! ([`in_flight`], [`expected_in_flight`]).
//! One while a command buffer not yet committed used it detaches on both, an unnamed reach
//! reading 0 and the holder failing with code 8 ([`held`]). An object detached at once and never
//! replaced reads 0 on both where a kernel reaches it unnamed, its `contents` and the memory it
//! was made over staying, and a buffer's deallocator running only once it went
//! ([`never_replaced`]).

#![cfg(target_os = "macos")] // Compares with Metal, which is macOS's.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs,
    unsafe_code
)]

use std::os::unix::process::ExitStatusExt as _;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::{EventRef, SharedEvent};
use reims_vgpu_core::platform::os::clock::{Clock, MonoTime};
use reims_vgpu_core::platform::os::vm::{HostVm, VmProt};
use reims_vgpu_core::platform::resource::{Buffer, Heap, Library, Texture};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize, NsRange, ResourceAddressRange};
use reims_vgpu_core::platform::values::{BufferDesc, HeapDesc, OptionSet, TextureDesc};
use reims_vgpu_core::platform::{HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::tasks::launcher::{HostSide, ServiceLink, TaskLauncher};
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_metal::os::vm::MetalVm;
use reims_vgpu_metal::spi::methods::{BufferSpi, HeapSpi};
use reims_vgpu_vulkan::VulkanPlatform;

const PAGE: u64 = 0x4000;

/// A launcher no test launches through.
struct NoLauncher;

impl<P: Platform> TaskLauncher<P> for NoLauncher {
    fn launch(&self, _: &Arc<P>, _: Arc<dyn HostSide<P>>) -> Arc<dyn ServiceLink<P>> {
        panic!("no task is launched in these tests")
    }
}

/// Fresh pages of the host, each byte `byte`.
fn pages(count: u64, byte: u8) -> HostPtr {
    let at = MetalVm::new().allocate_anywhere(count * PAGE).unwrap();
    // SAFETY: the pages were just allocated, readable and writable.
    unsafe {
        std::ptr::write_bytes(
            at.as_ptr::<u8>(),
            byte,
            usize::try_from(count * PAGE).unwrap(),
        );
    };
    at
}

fn read(at: HostPtr, offset: u64) -> u8 {
    // SAFETY: the test's own mapped memory, or a buffer's contents over it.
    unsafe { at.wrapping_add(offset).as_ptr::<u8>().read_volatile() }
}

fn write(at: HostPtr, offset: u64, byte: u8) {
    // SAFETY: the test's own mapped, writable memory.
    unsafe { at.wrapping_add(offset).as_ptr::<u8>().write_volatile(byte) }
}

/// Word `index` of the 32-bit words at `at`.
fn word(at: HostPtr, index: usize) -> u32 {
    // SAFETY: the test's own mapped memory.
    unsafe { at.as_ptr::<u32>().add(index).read_volatile() }
}

/// Waits past the time Metal takes to retire an object's last GPU use: a detach before then
/// answers NO and leaves the object attached (measured: 9 of 200 detaches right after
/// `waitUntilCompleted`, none 2 ms later; `src/gpu/backing.rs`'s docs). Every detach of an object
/// a completed command buffer used comes after it.
fn retire() {
    std::thread::sleep(std::time::Duration::from_millis(20));
}

fn range(at: HostPtr, offset: u64, length: u64) -> ResourceAddressRange {
    ResourceAddressRange {
        address: u64::try_from(at.addr()).unwrap() + offset,
        length,
    }
}

/// What the platform does, step by step: each step's name and what it answered or showed.
type Steps = Vec<(&'static str, Vec<u64>)>;

/// A GPU fill of the buffer's first `length` bytes with `value`, then a copy of its bytes at
/// `offsets` out to a buffer of the device's: the command buffer's status, and the bytes.
fn fill_then_copy<P: Platform>(
    dev: &P::Device,
    buffer: &P::Buffer,
    length: u64,
    value: Option<u8>,
    offsets: &[u64],
) -> Vec<u64> {
    let out = dev
        .new_buffer_of_length(OnFault::Return, length, 0)
        .unwrap()
        .unwrap();
    let cb = dev
        .new_command_queue()
        .unwrap()
        .command_buffer(OnFault::Return)
        .unwrap()
        .unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    if let Some(value) = value {
        let fill = BlitCmd::FillBuffer {
            buffer,
            range: NsRange::new(0, length),
            value,
        };
        e.encode(OnFault::Return, fill).unwrap();
    }
    let copy = BlitCmd::CopyBufferToBuffer {
        source_buffer: Some(buffer),
        source_offset: 0,
        destination_buffer: Some(&out),
        destination_offset: 0,
        size: length,
    };
    e.encode(OnFault::Return, copy).unwrap();
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    let mut seen = vec![cb.status().0];
    seen.extend(
        offsets
            .iter()
            .map(|&at| u64::from(read(out.contents(), at))),
    );
    seen
}

/// `seen`, a [`fill_then_copy`] of `fill` into read-only pages of `pages` bytes: the status, then
/// each byte the copy read as the fill where it read the pages. The copy later in the fill's
/// command buffer reads the dropped fill while the GPU's caches hold it, and the pages where
/// they no longer do, which is Metal's own variance: Metal reads the fill when nothing else runs
/// on the GPU, and the pages where 64 MiB of other GPU work came between
/// (`tests/client_direction.rs`, `BufferFillThrash`), and this platform's fill and copy reach
/// Metal's GPU through a read-only mapping too. Any other byte stays, and fails the comparison.
fn cached_or_pages(mut seen: Vec<u64>, fill: u8, pages: u8) -> Vec<u64> {
    for byte in seen.iter_mut().skip(1) {
        if *byte == u64::from(pages) {
            *byte = u64::from(fill);
        }
    }
    seen
}

/// A detached buffer of three pages: made over a temporary allocation, detached, replaced by
/// scattered guest pages, detached and replaced by other pages read-only, then writable.
fn buffer_life<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let mut steps: Steps = Vec::new();
    let length = 3 * PAGE;
    let dummy = pages(3, 0x11);
    let (ran, runs) = mpsc::channel();
    // SAFETY: the dummy's pages stay mapped until the deallocator, which frees them, runs.
    let contents = unsafe { HostRange::new(dummy, length) };
    let buffer = dev
        .new_buffer(BufferDesc {
            length,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: Some(contents),
            no_copy: true,
            deallocator: Some(Box::new(move |at, length| {
                let _ = ran.send(at == dummy);
                let _ = MetalVm::new().deallocate(at, length);
            })),
        })
        .unwrap()
        .unwrap();
    let gpu_address = buffer.gpu_address();
    buffer.detach_backing().unwrap();
    let guest = pages(5, 0xa1);
    write(guest, 3 * PAGE, 0xb2);
    let scattered = [
        range(guest, 0, PAGE),
        range(guest, PAGE, PAGE),
        range(guest, 3 * PAGE, PAGE),
    ];
    let null = [
        range(guest, 0, PAGE),
        range(guest, PAGE, PAGE),
        ResourceAddressRange {
            address: 0,
            length: PAGE,
        },
    ];
    let mut answers = Vec::new();
    for ranges in [
        &[][..],
        &null[..],
        &scattered[..2],
        &[&scattered[..], &[range(guest, 4 * PAGE, PAGE)]].concat()[..],
    ] {
        let array = p.new_address_range_array(ranges).unwrap();
        answers.push(u64::from(buffer.replace_backing(&array, false).unwrap()));
    }
    steps.push(("refused replaces", answers));
    steps.push((
        "deallocator runs",
        vec![u64::try_from(runs.try_iter().count()).unwrap()],
    ));
    let array = p.new_address_range_array(&scattered).unwrap();
    steps.push((
        "replace, then again",
        vec![
            u64::from(buffer.replace_backing(&array, false).unwrap()),
            u64::from(buffer.replace_backing(&array, false).unwrap()),
        ],
    ));
    steps.push((
        "the dummy's deallocator ran",
        runs.try_iter().map(u64::from).collect(),
    ));
    steps.push((
        "same GPU address",
        vec![u64::from(buffer.gpu_address() == gpu_address)],
    ));
    let contents = buffer.contents();
    steps.push((
        "contents",
        [0, PAGE, 2 * PAGE]
            .map(|at| u64::from(read(contents, at)))
            .to_vec(),
    ));
    steps.push((
        "the GPU's fill",
        fill_then_copy::<P>(dev, &buffer, length, Some(0x5c), &[0, 2 * PAGE]),
    ));
    steps.push((
        "the guest's pages",
        [0, PAGE, 2 * PAGE, 3 * PAGE, 4 * PAGE]
            .map(|at| u64::from(read(guest, at)))
            .to_vec(),
    ));
    write(guest, 3 * PAGE + 8, 0x6d);
    steps.push((
        "the GPU reads the guest's write",
        fill_then_copy::<P>(dev, &buffer, length, None, &[2 * PAGE + 8]),
    ));
    retire();
    buffer.detach_backing().unwrap();
    buffer.detach_backing().unwrap();
    steps.push((
        "contents stays while detached",
        vec![u64::from(buffer.contents() == contents)],
    ));
    let other = pages(3, 0xc3);
    for read_only in [true, false] {
        let array = p
            .new_address_range_array(&[range(other, 0, length)])
            .unwrap();
        steps.push((
            "replace by other pages",
            vec![u64::from(
                buffer.replace_backing(&array, read_only).unwrap(),
            )],
        ));
        let value = if read_only { 0x7e } else { 0x7f };
        let seen = fill_then_copy::<P>(dev, &buffer, length, Some(value), &[0, 2 * PAGE]);
        steps.push((
            "the GPU's fill",
            match read_only {
                true => cached_or_pages(seen, value, 0xc3),
                false => seen,
            },
        ));
        steps.push((
            "the other pages",
            vec![u64::from(read(other, 0)), u64::from(read(other, 2 * PAGE))],
        ));
        steps.push((
            "the GPU's next read",
            fill_then_copy::<P>(dev, &buffer, length, None, &[0, 2 * PAGE]),
        ));
        retire();
        buffer.detach_backing().unwrap();
    }
    steps.push((
        "the first pages",
        vec![u64::from(read(guest, 0)), u64::from(read(guest, 3 * PAGE))],
    ));
    drop(buffer);
    steps
}

/// A detached heap of two pages: made over a temporary allocation's address range, detached,
/// with a buffer over all of it, replaced by scattered guest pages (the allocation then freed, as
/// core frees it), detached and replaced by other pages read-only.
fn heap_life<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let mut steps: Steps = Vec::new();
    let size = 2 * PAGE;
    let vm = MetalVm::new();
    let dummy = pages(2, 0);
    let array = p.new_address_range_array(&[range(dummy, 0, size)]).unwrap();
    let made = dev
        .new_heap(&HeapDesc {
            size,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            address_ranges: Some(&array),
            pinned_gpu_address: None,
        })
        .unwrap();
    let heap = made.heap.unwrap();
    let gpu_address = heap.gpu_address();
    heap.detach_backing().unwrap();
    let cover = heap
        .new_buffer_of_length(size, made.descriptor_resource_options, 0)
        .unwrap()
        .unwrap();
    let guest = pages(3, 0xa1);
    let scattered = [range(guest, 0, PAGE), range(guest, 2 * PAGE, PAGE)];
    let mut answers = Vec::new();
    for ranges in [
        &scattered[..1],
        &[range(guest, 0, PAGE), range(guest, PAGE, 2 * PAGE)][..],
        &[
            range(guest, 0, PAGE),
            ResourceAddressRange {
                address: 0,
                length: PAGE,
            },
        ][..],
        &scattered[..],
        &scattered[..],
    ] {
        let array = p.new_address_range_array(ranges).unwrap();
        answers.push(u64::from(heap.replace_backing(&array, false).unwrap()));
    }
    steps.push(("replaces", answers));
    assert!(vm.deallocate(dummy, size).is_success());
    steps.push((
        "same GPU address",
        vec![u64::from(heap.gpu_address() == gpu_address)],
    ));
    steps.push((
        "the GPU's fill",
        fill_then_copy::<P>(dev, &cover, size, Some(0x5c), &[0, PAGE]),
    ));
    steps.push((
        "the guest's pages",
        [0, PAGE, 2 * PAGE]
            .map(|at| u64::from(read(guest, at)))
            .to_vec(),
    ));
    write(guest, 2 * PAGE + 4, 0x33);
    steps.push((
        "the GPU reads the guest's write",
        fill_then_copy::<P>(dev, &cover, size, None, &[PAGE + 4]),
    ));
    retire();
    heap.detach_backing().unwrap();
    let other = pages(2, 0xc3);
    let array = p.new_address_range_array(&[range(other, 0, size)]).unwrap();
    steps.push((
        "read-only replace",
        vec![u64::from(heap.replace_backing(&array, true).unwrap())],
    ));
    let seen = fill_then_copy::<P>(dev, &cover, size, Some(0x7e), &[0, PAGE]);
    steps.push(("the GPU's fill", cached_or_pages(seen, 0x7e, 0xc3)));
    steps.push((
        "the other pages",
        vec![u64::from(read(other, 0)), u64::from(read(other, PAGE))],
    ));
    steps.push((
        "the GPU's next read",
        fill_then_copy::<P>(dev, &cover, size, None, &[0, PAGE]),
    ));
    steps
}

/// Heaps over address ranges Metal makes or answers nil for, and a no-copy buffer of length 0,
/// which is nil and whose deallocator never runs: 1 for each made.
fn edges<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let guest = pages(4, 0xa1);
    let null = ResourceAddressRange {
        address: 0,
        length: 0,
    };
    let heaps = [
        (2 * PAGE, vec![range(guest, 0, PAGE)]),
        (PAGE, vec![range(guest, 0, 2 * PAGE)]),
        (0, vec![null]),
        (PAGE, vec![]),
        (
            PAGE,
            vec![ResourceAddressRange {
                address: 0,
                length: PAGE,
            }],
        ),
    ]
    .map(|(size, ranges)| {
        let array = p.new_address_range_array(&ranges).unwrap();
        let made = dev
            .new_heap(&HeapDesc {
                size,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0)],
                address_ranges: Some(&array),
                pinned_gpu_address: None,
            })
            .unwrap();
        u64::from(made.heap.is_some())
    });
    let (ran, runs) = mpsc::channel();
    // SAFETY: no byte of a range of length 0 is read.
    let contents = unsafe { HostRange::new(HostPtr::NULL, 0) };
    let buffer = dev
        .new_buffer(BufferDesc {
            length: 0,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: Some(contents),
            no_copy: true,
            deallocator: Some(Box::new(move |_, _| {
                let _ = ran.send(());
            })),
        })
        .unwrap();
    vec![
        ("heaps", heaps.to_vec()),
        (
            "a buffer of length 0, its deallocator's runs",
            vec![
                u64::from(buffer.is_some()),
                u64::try_from(runs.try_iter().count()).unwrap(),
            ],
        ),
    ]
}

/// A detached buffer of `count` pages, as core makes one: a no-copy buffer over a temporary
/// allocation, detached at once.
fn detached_buffer<P: Platform>(dev: &P::Device, count: u64) -> P::Buffer {
    let length = count * PAGE;
    let dummy = pages(count, 0);
    // SAFETY: the dummy's pages stay mapped until the deallocator, which frees them, runs.
    let contents = unsafe { HostRange::new(dummy, length) };
    let buffer = dev
        .new_buffer(BufferDesc {
            length,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: Some(contents),
            no_copy: true,
            deallocator: Some(Box::new(|at, length| {
                let _ = MetalVm::new().deallocate(at, length);
            })),
        })
        .unwrap()
        .unwrap();
    buffer.detach_backing().unwrap();
    buffer
}

/// A Shared buffer of the device's, of one page of zeros.
fn plain_buffer<P: Platform>(dev: &P::Device) -> P::Buffer {
    let buffer = dev
        .new_buffer_of_length(OnFault::Return, PAGE, 0)
        .unwrap()
        .unwrap();
    // SAFETY: the buffer's own Shared contents, one page.
    unsafe { std::ptr::write_bytes(buffer.contents().as_ptr::<u8>(), 0, PAGE as usize) };
    buffer
}

/// A blit encoder of `cb` with `commands`, ended.
fn blit<P: Platform>(cb: &P::CommandBuffer, commands: Vec<BlitCmd<'_, P>>) {
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    for command in commands {
        e.encode(OnFault::Return, command).unwrap();
    }
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
}

/// A fill of `buffer`'s 16 bytes at `at` with `value`.
fn fill<P: Platform>(buffer: &P::Buffer, at: u64, value: u8) -> BlitCmd<'_, P> {
    BlitCmd::FillBuffer {
        buffer,
        range: NsRange::new(at, 16),
        value,
    }
}

/// A copy of `texture`'s first row of 16 pixels to the start of `buffer`.
fn copy_out<'a, P: Platform>(texture: &'a P::Texture, buffer: &'a P::Buffer) -> BlitCmd<'a, P> {
    BlitCmd::CopyTextureToBuffer {
        source_texture: texture,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(16, 1, 1),
        destination_buffer: buffer,
        destination_offset: 0,
        destination_bytes_per_row: 256,
        destination_bytes_per_image: 256,
    }
}

/// A 2D RGBA8Unorm texture of 64 by 4 pixels, Shared, rows of 256 bytes.
fn texture_desc<P: Platform>(p: &P) -> P::TextureDescriptor {
    p.texture_descriptor(&TextureDesc {
        width: 64,
        height: 4,
        resource_options: Some(0),
        ..TextureDesc::default()
    })
}

/// `cb` committed and waited for: its status and its error's code (0 for none).
fn finish<P: Platform>(cb: &P::CommandBuffer) -> Vec<u64> {
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    vec![
        cb.status().0,
        cb.error().map_or(0, |e| u64::try_from(e.code).unwrap()),
    ]
}

/// A command buffer of `queue`.
fn command_buffer<P: Platform>(queue: &P::CommandQueue) -> P::CommandBuffer {
    queue.command_buffer(OnFault::Return).unwrap().unwrap()
}

/// Command buffers that use a detached buffer or heap: what runs of them, what they signal and
/// report, and where the guest's pages change.
fn detached_use<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let mut steps: Steps = Vec::new();
    let queue = dev.new_command_queue().unwrap();
    let byte = |buffer: &P::Buffer, at: u64| u64::from(read(buffer.contents(), at));

    // One submission: a fill of another buffer before the detached one's, in one encoder.
    let detached = detached_buffer::<P>(dev, 1);
    let other = plain_buffer::<P>(dev);
    let signaled = dev.new_shared_event().unwrap();
    let scheduled = dev.new_shared_event().unwrap();
    let cb = command_buffer::<P>(&queue);
    let (seen, handler) = mpsc::channel();
    cb.add_completed_handler(
        OnFault::Return,
        Box::new(move |info| {
            let code = info.error.map_or(0, |e| u64::try_from(e.code).unwrap());
            let _ = seen.send(vec![info.status.0, code]);
        }),
    )
    .unwrap();
    cb.encode_signal_when_scheduled(Some(&scheduled), 1)
        .unwrap();
    blit::<P>(
        &cb,
        vec![fill::<P>(&other, 0, 0x11), fill::<P>(&detached, 0, 0x5a)],
    );
    cb.encode_signal(Some(EventRef::Shared(&signaled)), 3)
        .unwrap();
    let mut seen = finish::<P>(&cb);
    seen.push(byte(&other, 0));
    seen.push(u64::from(signaled.wait_until_signaled_value(3, 1000)));
    seen.push(u64::from(scheduled.wait_until_signaled_value(1, 1000)));
    seen.extend(handler.recv().unwrap());
    steps.push(("one submission", seen));

    // A texture over the detached buffer, and a view of it.
    let texture = detached
        .new_texture(OnFault::Return, &texture_desc(p), 0, 256)
        .unwrap()
        .unwrap();
    let view = texture.new_texture_view(70).unwrap().unwrap();
    let mut seen = Vec::new();
    for texture in [&texture, &view] {
        let cb = command_buffer::<P>(&queue);
        blit::<P>(&cb, vec![copy_out::<P>(texture, &other)]);
        seen.extend(finish::<P>(&cb));
    }
    steps.push(("a texture over it, a view of that", seen));

    // Replaced, it runs.
    let guest = pages(1, 0x01);
    let array = p.new_address_range_array(&[range(guest, 0, PAGE)]).unwrap();
    assert!(detached.replace_backing(&array, false).unwrap());
    let cb = command_buffer::<P>(&queue);
    blit::<P>(&cb, vec![fill::<P>(&detached, 0, 0x5a)]);
    let mut seen = finish::<P>(&cb);
    seen.push(u64::from(read(guest, 0)));
    steps.push(("replaced", seen));

    // A buffer and a texture placed in a detached heap, then the heap replaced.
    let size = 4 * PAGE;
    let dummy = pages(4, 0);
    let array = p.new_address_range_array(&[range(dummy, 0, size)]).unwrap();
    let made = dev
        .new_heap(&HeapDesc {
            size,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            address_ranges: Some(&array),
            pinned_gpu_address: None,
        })
        .unwrap();
    let heap = made.heap.unwrap();
    let placed = heap
        .new_buffer_of_length(PAGE, made.descriptor_resource_options, 0)
        .unwrap()
        .unwrap();
    let placed_texture = heap
        .new_texture(&texture_desc(p), 2 * PAGE)
        .unwrap()
        .unwrap();
    heap.detach_backing().unwrap();
    let mut seen = Vec::new();
    for commands in [
        vec![fill::<P>(&placed, 0, 0x44)],
        vec![copy_out::<P>(&placed_texture, &other)],
    ] {
        let cb = command_buffer::<P>(&queue);
        blit::<P>(&cb, commands);
        seen.extend(finish::<P>(&cb));
    }
    let heap_guest = pages(4, 0x02);
    let array = p
        .new_address_range_array(&[range(heap_guest, 0, size)])
        .unwrap();
    assert!(heap.replace_backing(&array, false).unwrap());
    assert!(vm_free(dummy, size));
    let cb = command_buffer::<P>(&queue);
    blit::<P>(&cb, vec![fill::<P>(&placed, 0, 0x44)]);
    seen.extend(finish::<P>(&cb));
    seen.push(u64::from(read(heap_guest, 0)));
    steps.push(("placed in a detached heap, then replaced", seen));

    // Another submission, past an event signal on each side of the detached buffer's encoder:
    // both run.
    let detached = detached_buffer::<P>(dev, 1);
    let other = plain_buffer::<P>(dev);
    let signaled = dev.new_shared_event().unwrap();
    let cb = command_buffer::<P>(&queue);
    blit::<P>(&cb, vec![fill::<P>(&other, 0, 0x11)]);
    cb.encode_signal(Some(EventRef::Shared(&signaled)), 1)
        .unwrap();
    blit::<P>(&cb, vec![fill::<P>(&detached, 0, 0x5a)]);
    cb.encode_signal(Some(EventRef::Shared(&signaled)), 2)
        .unwrap();
    blit::<P>(&cb, vec![fill::<P>(&other, 16, 0x22)]);
    let mut seen = finish::<P>(&cb);
    seen.extend([byte(&other, 0), byte(&other, 16)]);
    steps.push(("other submissions", seen));

    // Released detached after its use is encoded, before the commit.
    let detached = detached_buffer::<P>(dev, 1);
    let other = plain_buffer::<P>(dev);
    let cb = command_buffer::<P>(&queue);
    blit::<P>(
        &cb,
        vec![fill::<P>(&other, 0, 0x11), fill::<P>(&detached, 0, 0x5a)],
    );
    drop(detached);
    let mut seen = finish::<P>(&cb);
    seen.push(byte(&other, 0));
    steps.push(("released before the commit", seen));

    // A texture over a buffer released detached, used by a command buffer made after the
    // release: it fails.
    let detached = detached_buffer::<P>(dev, 1);
    let texture = detached
        .new_texture(OnFault::Return, &texture_desc(p), 0, 256)
        .unwrap()
        .unwrap();
    drop(detached);
    let cb = command_buffer::<P>(&queue);
    blit::<P>(&cb, vec![copy_out::<P>(&texture, &other)]);
    steps.push((
        "a texture over a buffer released detached",
        finish::<P>(&cb),
    ));
    steps
}

/// The tests' kernel that reads the word at an address it loads, and writes the word after it
/// (`src/gpu/reach.metal`).
const REACH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reach.metallib"));

/// The reach of `buffer` through an address the kernel loads, naming nothing of it: the command
/// buffer's status and error code, the word read, and whether the guest's word at `written`
/// took the write.
fn reach<P: Platform>(
    dev: &P::Device,
    queue: &P::CommandQueue,
    buffer: &P::Buffer,
    written: HostPtr,
) -> Vec<u64> {
    reach_at::<P>(dev, queue, buffer, 0, written)
}

/// [`reach`] of `buffer`'s bytes from `offset` on: the kernel reads the word there and writes
/// the one after it, whose host word at `written` (plus 4) shows whether it took the write.
fn reach_at<P: Platform>(
    dev: &P::Device,
    queue: &P::CommandQueue,
    buffer: &P::Buffer,
    offset: u64,
    written: HostPtr,
) -> Vec<u64> {
    let library = dev.new_library(REACH).unwrap().unwrap().unwrap();
    let function = library
        .new_function(OnFault::Return, "reach")
        .unwrap()
        .unwrap();
    let pipeline = dev
        .new_compute_pipeline_state_for_function(Some(&function))
        .unwrap()
        .unwrap();
    let addresses = plain_buffer::<P>(dev);
    let words = plain_buffer::<P>(dev);
    // SAFETY: both are the device's own Shared buffers of a page.
    unsafe {
        addresses
            .contents()
            .as_ptr::<u64>()
            .write(buffer.gpu_address() + offset);
        words.contents().as_ptr::<u32>().add(1).write(MARKER);
    }
    let cb = command_buffer::<P>(queue);
    let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
    for command in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&pipeline),
        },
        ComputeCmd::SetBuffers {
            buffers: &[Some(&addresses), Some(&words)],
            offsets: &[0, 0],
            range: NsRange::new(0, 2),
        },
        ComputeCmd::DispatchThreadgroups {
            threadgroups_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Return, command).unwrap();
    }
    let mut seen = finish::<P>(&cb);
    // SAFETY: the words buffer's own contents, and the guest's mapped page.
    let (read, landed) = unsafe {
        (
            words.contents().as_ptr::<u32>().read_volatile(),
            written.as_ptr::<u32>().add(1).read_volatile(),
        )
    };
    seen.extend([u64::from(read), u64::from(landed == MARKER)]);
    seen
}

/// The word [`reach`] writes.
const MARKER: u32 = 0xc0de_5eed;

/// A buffer, and a buffer placed in a heap, over the guest's pages and used, reached through an
/// address a kernel loads, attached, then detached, then replaced by new pages, before and after
/// a use: what each platform does where nothing names the buffer.
fn detached_reach<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let mut steps: Steps = Vec::new();
    let queue = dev.new_command_queue().unwrap();
    let use_ = |buffer: &P::Buffer| {
        let cb = command_buffer::<P>(&queue);
        blit::<P>(&cb, vec![fill::<P>(buffer, 256, 0x77)]);
        finish::<P>(&cb)
    };

    let buffer = detached_buffer::<P>(dev, 1);
    let guest = pages(1, 0xa1);
    let array = p.new_address_range_array(&[range(guest, 0, PAGE)]).unwrap();
    assert!(buffer.replace_backing(&array, false).unwrap());
    let mut seen = use_(&buffer);
    seen.extend(reach::<P>(dev, &queue, &buffer, guest));
    for offset in 4..8 {
        write(guest, offset, 0xa1);
    }
    retire();
    let contents = buffer.contents();
    buffer.detach_backing().unwrap();
    seen.extend(reach::<P>(dev, &queue, &buffer, guest));
    seen.push(u64::from(buffer.contents() == contents));
    steps.push(("a buffer", seen));
    let fresh = pages(1, 0xb1);
    let array = p.new_address_range_array(&[range(fresh, 0, PAGE)]).unwrap();
    assert!(buffer.replace_backing(&array, false).unwrap());
    let mut seen = reach::<P>(dev, &queue, &buffer, fresh);
    seen.push(u64::from(word(guest, 1) == MARKER));
    seen.extend(use_(&buffer));
    seen.extend(reach::<P>(dev, &queue, &buffer, fresh));
    steps.push(("a buffer used, detached, replaced", seen));

    let size = 4 * PAGE;
    let dummy = pages(4, 0);
    let array = p.new_address_range_array(&[range(dummy, 0, size)]).unwrap();
    let made = dev
        .new_heap(&HeapDesc {
            size,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            address_ranges: Some(&array),
            pinned_gpu_address: None,
        })
        .unwrap();
    let heap = made.heap.unwrap();
    let second = heap
        .new_buffer_of_length(PAGE, made.descriptor_resource_options, 2 * PAGE)
        .unwrap()
        .unwrap();
    heap.detach_backing().unwrap();
    let guest = pages(4, 0xa2);
    let array = p.new_address_range_array(&[range(guest, 0, size)]).unwrap();
    assert!(heap.replace_backing(&array, false).unwrap());
    assert!(vm_free(dummy, size));
    let at = guest.wrapping_add(2 * PAGE);
    let mut seen = use_(&second);
    seen.extend(reach::<P>(dev, &queue, &second, at));
    for offset in 4..8 {
        write(at, offset, 0xa2);
    }
    retire();
    let contents = second.contents();
    heap.detach_backing().unwrap();
    seen.extend(reach::<P>(dev, &queue, &second, at));
    seen.push(u64::from(second.contents() == contents));
    steps.push(("a buffer placed in a heap", seen));
    let fresh = pages(4, 0xb2);
    let array = p.new_address_range_array(&[range(fresh, 0, size)]).unwrap();
    assert!(heap.replace_backing(&array, false).unwrap());
    let new_at = fresh.wrapping_add(2 * PAGE);
    let mut seen = reach::<P>(dev, &queue, &second, new_at);
    seen.push(u64::from(word(at, 1) == MARKER));
    seen.extend(use_(&second));
    seen.extend(reach::<P>(dev, &queue, &second, new_at));
    steps.push(("a heap used, detached, replaced", seen));
    steps
}

/// What both platforms do in [`detached_reach`]: attached, the kernel reads the guest's word
/// and its write lands there; detached, it reads 0 and its write is dropped, the command buffer
/// completing with no error; `contents` stays. Replaced after the detach, the kernel reads 0
/// and its write lands neither in the new pages nor in the old, until a command buffer uses the
/// buffer, or a resource of its heap, again; then it reads and writes the new pages.
fn expected_reach() -> Steps {
    vec![
        ("a buffer", vec![4, 0, 4, 0, 0xa1a1_a1a1, 1, 4, 0, 0, 0, 1]),
        (
            "a buffer used, detached, replaced",
            vec![4, 0, 0, 0, 0, 4, 0, 4, 0, 0xb1b1_b1b1, 1],
        ),
        (
            "a buffer placed in a heap",
            vec![4, 0, 4, 0, 0xa2a2_a2a2, 1, 4, 0, 0, 0, 1],
        ),
        (
            "a heap used, detached, replaced",
            vec![4, 0, 0, 0, 0, 4, 0, 4, 0, 0xb2b2_b2b2, 1],
        ),
    ]
}

/// Buffers placed in a heap over address ranges, and in a plain placement heap: one's detach,
/// then fills of it and of another of the heap, its replace, and the heap's detach; then a
/// buffer, and a texture, each placed in a heap of its own released detached, used by a command
/// buffer made after the release.
fn placed_detach<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let mut steps: Steps = Vec::new();
    let queue = dev.new_command_queue().unwrap();
    let fill_one = |buffer: &P::Buffer, value: u8| {
        let cb = command_buffer::<P>(&queue);
        blit::<P>(&cb, vec![fill::<P>(buffer, 0, value)]);
        finish::<P>(&cb)
    };
    let size = 4 * PAGE;
    let heap_over = |over_ranges: bool| {
        let memory = pages(4, 0x31);
        let array = p
            .new_address_range_array(&[range(memory, 0, size)])
            .unwrap();
        let made = dev
            .new_heap(&HeapDesc {
                size,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: over_ranges.then_some(&array),
                pinned_gpu_address: None,
            })
            .unwrap();
        (made.heap.unwrap(), made.descriptor_resource_options)
    };
    for (name, over_ranges) in [
        ("placed in a heap over address ranges", true),
        ("placed in a plain placement heap", false),
    ] {
        let (heap, options) = heap_over(over_ranges);
        let first = heap
            .new_buffer_of_length(PAGE, options, 0)
            .unwrap()
            .unwrap();
        let second = heap
            .new_buffer_of_length(PAGE, options, 2 * PAGE)
            .unwrap()
            .unwrap();
        let mut seen = fill_one(&first, 0x11);
        retire();
        first.detach_backing().unwrap();
        seen.extend(fill_one(&first, 0x22));
        seen.extend(fill_one(&second, 0x33));
        let guest = pages(1, 0x01);
        let array = p.new_address_range_array(&[range(guest, 0, PAGE)]).unwrap();
        seen.push(u64::from(first.replace_backing(&array, false).unwrap()));
        seen.extend(fill_one(&first, 0x44));
        heap.detach_backing().unwrap();
        seen.extend(fill_one(&second, 0x55));
        steps.push((name, seen));
    }
    // Each in a heap of its own, so that the texture's heap is held by nothing else.
    let (heap, options) = heap_over(true);
    let buffer = heap
        .new_buffer_of_length(PAGE, options, 0)
        .unwrap()
        .unwrap();
    heap.detach_backing().unwrap();
    drop(heap);
    let (heap, _) = heap_over(true);
    let texture = heap.new_texture(&texture_desc(p), 0).unwrap().unwrap();
    heap.detach_backing().unwrap();
    drop(heap);
    let other = plain_buffer::<P>(dev);
    let mut seen = fill_one(&buffer, 0x66);
    let cb = command_buffer::<P>(&queue);
    blit::<P>(&cb, vec![copy_out::<P>(&texture, &other)]);
    seen.extend(finish::<P>(&cb));
    steps.push((
        "a buffer, and a texture, placed in a heap released detached",
        seen,
    ));
    steps
}

/// What both platforms do in [`placed_detach`].
fn expected_placed() -> Steps {
    vec![
        (
            "placed in a heap over address ranges",
            vec![4, 0, 5, 8, 5, 8, 0, 5, 8, 5, 8],
        ),
        (
            "placed in a plain placement heap",
            vec![4, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0],
        ),
        (
            "a buffer, and a texture, placed in a heap released detached",
            vec![5, 8, 5, 8],
        ),
    ]
}

/// Frees `length` bytes of the test's pages at `at`.
fn vm_free(at: HostPtr, length: u64) -> bool {
    MetalVm::new().deallocate(at, length).is_success()
}

/// What both platforms do in [`detached_use`].
fn expected_use() -> Steps {
    vec![
        ("one submission", vec![5, 8, 0, 1, 1, 5, 8]),
        ("a texture over it, a view of that", vec![5, 8, 5, 8]),
        ("replaced", vec![4, 0, 0x5a]),
        (
            "placed in a detached heap, then replaced",
            vec![5, 8, 5, 8, 4, 0, 0x44],
        ),
        ("other submissions", vec![5, 8, 0x11, 0x22]),
        ("released before the commit", vec![5, 8, 0]),
        ("a texture over a buffer released detached", vec![5, 8]),
    ]
}

/// What both platforms do ([`buffer_life`], [`heap_life`], [`edges`]).
fn expected() -> Vec<Steps> {
    vec![
        vec![
            ("refused replaces", vec![0, 0, 0, 0]),
            ("deallocator runs", vec![0]),
            ("replace, then again", vec![1, 0]),
            ("the dummy's deallocator ran", vec![1]),
            ("same GPU address", vec![1]),
            ("contents", vec![0xa1, 0xa1, 0xb2]),
            ("the GPU's fill", vec![4, 0x5c, 0x5c]),
            ("the guest's pages", vec![0x5c, 0x5c, 0xa1, 0x5c, 0xa1]),
            ("the GPU reads the guest's write", vec![4, 0x6d]),
            ("contents stays while detached", vec![1]),
            ("replace by other pages", vec![1]),
            ("the GPU's fill", vec![4, 0x7e, 0x7e]),
            ("the other pages", vec![0xc3, 0xc3]),
            ("the GPU's next read", vec![4, 0xc3, 0xc3]),
            ("replace by other pages", vec![1]),
            ("the GPU's fill", vec![4, 0x7f, 0x7f]),
            ("the other pages", vec![0x7f, 0x7f]),
            ("the GPU's next read", vec![4, 0x7f, 0x7f]),
            ("the first pages", vec![0x5c, 0x5c]),
        ],
        vec![
            ("replaces", vec![0, 0, 0, 1, 0]),
            ("same GPU address", vec![1]),
            ("the GPU's fill", vec![4, 0x5c, 0x5c]),
            ("the guest's pages", vec![0x5c, 0xa1, 0x5c]),
            ("the GPU reads the guest's write", vec![4, 0x33]),
            ("read-only replace", vec![1]),
            ("the GPU's fill", vec![4, 0x7e, 0x7e]),
            ("the other pages", vec![0xc3, 0xc3]),
            ("the GPU's next read", vec![4, 0xc3, 0xc3]),
        ],
        vec![
            ("heaps", vec![1, 1, 0, 0, 0]),
            ("a buffer of length 0, its deallocator's runs", vec![0, 0]),
        ],
    ]
}

#[test]
fn detached_buffers_and_heaps_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    let seen = vec![buffer_life(&p, &dev), heap_life(&p, &dev), edges(&p, &dev)];
    assert_eq!(seen, expected());
    assert_eq!(detached_use(&p, &dev), expected_use());
    assert_eq!(placed_detach(&p, &dev), expected_placed());
    assert_eq!(detached_reach(&p, &dev), expected_reach());
}

#[test]
fn detached_buffers_and_heaps_on_vulkan() {
    let Some(dev) = VulkanPlatform::system_default_device() else {
        return;
    };
    let p = VulkanPlatform::new(Box::new(NoLauncher));
    let seen = vec![buffer_life(&p, &dev), heap_life(&p, &dev), edges(&p, &dev)];
    assert_eq!(seen, expected());
    assert_eq!(detached_use(&p, &dev), expected_use());
    assert_eq!(placed_detach(&p, &dev), expected_placed());
    assert_eq!(detached_reach(&p, &dev), expected_reach());
}

/// The cases of [`behind_a_wait`].
const BEHIND_A_WAIT: [&str; 5] = [
    "detached",
    "replaced",
    "detached, then replaced",
    "detached, used past a split",
    "detached, another replaced past a split",
];

/// Case `case` of [`BEHIND_A_WAIT`]: a buffer over guest pages (detached over a temporary
/// allocation, for `replaced`) whose fill a command buffer commits behind an event wait (past a
/// fill of another buffer and an event signal, which splits its submissions, for `used past a
/// split`), then detached, replaced by other pages, or both, before the wait's signal. For
/// `another replaced past a split`, the fill is followed by an event signal and a fill of a
/// second buffer over guest pages, and before the wait's signal the first buffer is detached and
/// the second detached and replaced by other pages. What shows: the command buffer's status and
/// error code, and the first byte of the guest's pages the first buffer was over and of those a
/// buffer was replaced by.
fn behind_a_wait<P: Platform>(p: &P, dev: &P::Device, case: &str) -> Vec<u64> {
    let queue = dev.new_command_queue().unwrap();
    let buffer = detached_buffer::<P>(dev, 1);
    let over = pages(1, 0x01);
    if case != "replaced" {
        let array = p.new_address_range_array(&[range(over, 0, PAGE)]).unwrap();
        assert!(buffer.replace_backing(&array, false).unwrap());
    }
    let another = case.contains("another").then(|| {
        let second = detached_buffer::<P>(dev, 1);
        let array = p
            .new_address_range_array(&[range(pages(1, 0x02), 0, PAGE)])
            .unwrap();
        assert!(second.replace_backing(&array, false).unwrap());
        second
    });
    let gate = dev.new_shared_event().unwrap();
    let cb = command_buffer::<P>(&queue);
    cb.encode_wait(Some(EventRef::Shared(&gate)), 1).unwrap();
    let other = plain_buffer::<P>(dev);
    if case.ends_with("used past a split") {
        let split = dev.new_shared_event().unwrap();
        blit::<P>(&cb, vec![fill::<P>(&other, 0, 0x11)]);
        cb.encode_signal(Some(EventRef::Shared(&split)), 1).unwrap();
    }
    blit::<P>(&cb, vec![fill::<P>(&buffer, 0, 0x5c)]);
    if let Some(second) = &another {
        let split = dev.new_shared_event().unwrap();
        cb.encode_signal(Some(EventRef::Shared(&split)), 1).unwrap();
        blit::<P>(&cb, vec![fill::<P>(second, 0, 0x5c)]);
    }
    cb.commit(OnFault::Return).unwrap();
    if case.starts_with("detached") {
        buffer.detach_backing().unwrap();
    }
    let new = pages(1, 0x03);
    if case.starts_with("replaced") || case.ends_with("then replaced") {
        let array = p.new_address_range_array(&[range(new, 0, PAGE)]).unwrap();
        assert!(buffer.replace_backing(&array, false).unwrap());
    }
    if let Some(second) = &another {
        second.detach_backing().unwrap();
        let array = p.new_address_range_array(&[range(new, 0, PAGE)]).unwrap();
        assert!(second.replace_backing(&array, false).unwrap());
    }
    gate.set_signaled_value(1).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    vec![
        cb.status().0,
        cb.error().map_or(0, |e| u64::try_from(e.code).unwrap()),
        u64::from(read(over, 0)),
        u64::from(read(new, 0)),
    ]
}

/// Metal's answers in [`behind_a_wait`] (measured on the M3 Max, macOS 26.7): it decides each
/// submission as it submits it, so it fails one that uses the buffer detached then (code 8),
/// running the others, and runs one that uses a buffer replaced then over the new pages, also
/// after a submission of the same command buffer failed.
fn expected_behind_a_wait() -> [Vec<u64>; 5] {
    [
        vec![5, 8, 0x01, 0x03],
        vec![4, 0, 0x01, 0x5c],
        vec![4, 0, 0x01, 0x5c],
        vec![5, 8, 0x01, 0x03],
        vec![5, 8, 0x01, 0x5c],
    ]
}

#[test]
fn behind_a_wait_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    let seen = BEHIND_A_WAIT.map(|case| behind_a_wait(&p, &dev, case));
    assert_eq!(seen, expected_behind_a_wait());
}

/// The case of [`BEHIND_A_WAIT`] a child copy of this test binary runs.
const CHILD_CASE: &str = "RVG_BEHIND_A_WAIT";

/// This crate in [`behind_a_wait`], each case in a child copy of this test binary under strict
/// mode (`REIMS_VGPU_STRICT=1`), where a death ends the process as in production, on whatever thread it comes: each case answers Metal's,
/// and nothing dies. A use detached behind the wait, in the first submission or a later one,
/// fails with Metal's code 8 (the runtime decides each submission after the submission check
/// and keeps the code the check gave). A use replaced behind it, or detached and then replaced,
/// runs over the new pages: the runtime encodes the command buffer's encoders again over the
/// rebound buffer before its first submission
#[test]
fn behind_a_wait_on_vulkan() {
    if let Ok(case) = std::env::var(CHILD_CASE) {
        reims_vgpu_core::testkit::capture::set_capture_enabled(false);
        let dev = VulkanPlatform::system_default_device().unwrap();
        let p = VulkanPlatform::new(Box::new(NoLauncher));
        println!("seen {:?}", behind_a_wait(&p, &dev, &case));
        std::process::exit(0);
    }
    if VulkanPlatform::system_default_device().is_none() {
        return;
    }
    for (case, metal) in BEHIND_A_WAIT.into_iter().zip(expected_behind_a_wait()) {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "behind_a_wait_on_vulkan", "--nocapture"])
            .args(["--test-threads=1"])
            .env(CHILD_CASE, case)
            .env("REIMS_VGPU_STRICT", "1")
            .output()
            .unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        let death = stderr
            .lines()
            .find(|l| l.starts_with("reims-vgpu: fatal: "));
        assert!(out.status.success() && death.is_none(), "{case}: {out:?}");
        assert!(
            stdout.contains(&format!("seen {metal:?}")),
            "{case}: {stdout}"
        );
    }
}

/// A platform whose detach a test reads the answer of: Metal's `detachBacking` BOOL, which the
/// platform's `detach_backing` drops.
trait Answers: Platform {
    /// A buffer's `detachBacking`.
    fn detach_buffer(buffer: &Self::Buffer) -> bool;
    /// A heap's `detachBacking`.
    fn detach_heap(heap: &Self::Heap) -> bool;
}

impl Answers for MetalPlatform {
    fn detach_buffer(buffer: &Self::Buffer) -> bool {
        BufferSpi::detachBacking(buffer.get())
    }

    fn detach_heap(heap: &Self::Heap) -> bool {
        HeapSpi::detachBacking(heap.get())
    }
}

/// The cases of [`behind_a_gpu_wait`].
const BEHIND_A_GPU_WAIT: [&str; 5] = [
    "detached",
    "replaced",
    "detached, then replaced",
    "heap detached",
    "heap detached, then replaced",
];

/// When [`behind_a_gpu_wait`]'s signal is committed: after the window, behind a short blit; or
/// behind [`SPIN`]'s kernel of [`ROUNDS`] rounds, before the waiting command buffer, as core's
/// paging buffer is before the exec behind it, or [`STAND`] after it, before the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    After,
    First,
    Between,
}

/// How long [`behind_a_gpu_wait`] lets the waiting command buffer stand, committed, before the
/// window's calls.
const STAND: Duration = Duration::from_millis(10);

/// How long [`behind_a_gpu_wait`] waits for the window's calls to return before it commits the
/// signal anyway: a call that waits for the waiting command buffer to complete would otherwise
/// never return.
const BLOCKED: Duration = Duration::from_millis(500);

/// A buffer over the guest's pages `over` as core makes one, of a page (detached at once over a
/// temporary allocation, and replaced by `over` where `replaced`), or, for `heap`, a buffer
/// placed at the start of a heap over four pages made the same way: the buffer, and the heap.
fn gpu_wait_target<P: Platform>(
    p: &P,
    dev: &P::Device,
    heap: bool,
    over: HostPtr,
    replaced: bool,
) -> (P::Buffer, Option<P::Heap>) {
    let size = if heap { 4 * PAGE } else { PAGE };
    let array = p.new_address_range_array(&[range(over, 0, size)]).unwrap();
    if !heap {
        let buffer = detached_buffer::<P>(dev, 1);
        if replaced {
            assert!(buffer.replace_backing(&array, false).unwrap());
        }
        return (buffer, None);
    }
    let dummy = pages(4, 0);
    let ranges = p.new_address_range_array(&[range(dummy, 0, size)]).unwrap();
    let made = dev
        .new_heap(&HeapDesc {
            size,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            address_ranges: Some(&ranges),
            pinned_gpu_address: None,
        })
        .unwrap();
    let options = made.descriptor_resource_options;
    let heap = made.heap.unwrap();
    let target = heap
        .new_buffer_of_length(PAGE, options, 0)
        .unwrap()
        .unwrap();
    heap.detach_backing().unwrap();
    assert!(heap.replace_backing(&array, false).unwrap());
    assert!(vm_free(dummy, size));
    (target, Some(heap))
}

/// Case `case` of [`BEHIND_A_GPU_WAIT`]: a buffer over guest pages, or a buffer placed in a heap
/// over them (detached over a temporary allocation, for `replaced`), whose fill a command buffer
/// on one queue commits behind a wait on a plain event (`newEvent`), which only the GPU
/// signals, from a command buffer on another queue committed as `signal` says ([`Signal`]).
/// With the waiting command buffer committed for [`STAND`], the window: the object is detached,
/// replaced by other pages, or both, on a thread of its own. A signal after the window is
/// committed once the window's calls returned or [`BLOCKED`] passed; then both complete. What
/// shows: the waiting command buffer's status as the window opened, and whether its
/// `waitUntilScheduled` had returned on another thread by then; the window's answers, and
/// whether the fill had landed in the guest's pages as they returned; then the waiting command
/// buffer's status and error code, and the first byte of the guest's pages the object was over
/// and of those it was replaced by.
fn behind_a_gpu_wait<P: Answers>(p: &P, dev: &P::Device, case: &str, order: Signal) -> Steps {
    let heap = case.starts_with("heap");
    let size = if heap { 4 * PAGE } else { PAGE };
    let over = pages(size / PAGE, 0x01);
    let new = pages(size / PAGE, 0x03);
    let (target, made) = gpu_wait_target(p, dev, heap, over, case != "replaced");
    let replacement = p.new_address_range_array(&[range(new, 0, size)]).unwrap();
    let (first, second) = (
        dev.new_command_queue().unwrap(),
        dev.new_command_queue().unwrap(),
    );
    let gate = dev.new_event(OnFault::Return).unwrap().unwrap();
    let other = plain_buffer::<P>(dev);
    let signal = command_buffer::<P>(&second);
    if order != Signal::After {
        // SAFETY: the buffer's own Shared contents, one page.
        unsafe { other.contents().as_ptr::<u32>().write_volatile(ROUNDS) };
        spin::<P>(dev, &signal, &other);
        signal
            .encode_signal(Some(EventRef::Event(&gate)), 1)
            .unwrap();
    }
    if order == Signal::First {
        signal.commit(OnFault::Return).unwrap();
    }
    let waiting = command_buffer::<P>(&first);
    waiting
        .encode_wait(Some(EventRef::Event(&gate)), 1)
        .unwrap();
    blit::<P>(&waiting, vec![fill::<P>(&target, 0, 0x5c)]);
    waiting.commit(OnFault::Return).unwrap();
    if order == Signal::Between {
        std::thread::sleep(STAND);
        signal.commit(OnFault::Return).unwrap();
    }
    let clock = p.clock();
    let (status_before, scheduled_first, answers, landed) = std::thread::scope(|s| {
        let scheduled = s.spawn(|| {
            waiting.wait_until_scheduled().unwrap();
            clock.now_mono()
        });
        std::thread::sleep(STAND);
        let status_before = waiting.status().0;
        let opened = clock.now_mono();
        let (returned, calls) = mpsc::channel();
        let (target, made, replacement) = (&target, &made, &replacement);
        let window = s.spawn(move || {
            let mut answers = Vec::new();
            if case != "replaced" {
                answers.push(match made {
                    Some(heap) => P::detach_heap(heap),
                    None => P::detach_buffer(target),
                });
            }
            if case.contains("replaced") {
                answers.push(match made {
                    Some(heap) => heap.replace_backing(replacement, false).unwrap(),
                    None => target.replace_backing(replacement, false).unwrap(),
                });
            }
            let landed = read(over, 0) == 0x5c;
            returned.send(()).unwrap();
            (answers, landed)
        });
        if order == Signal::After {
            let _ = calls.recv_timeout(BLOCKED);
            blit::<P>(&signal, vec![fill::<P>(&other, 0, 0x11)]);
            signal
                .encode_signal(Some(EventRef::Event(&gate)), 1)
                .unwrap();
            signal.commit(OnFault::Return).unwrap();
        }
        let (answers, landed) = window.join().unwrap();
        let scheduled_first = scheduled.join().unwrap().as_nanos() < opened.as_nanos();
        (status_before, scheduled_first, answers, landed)
    });
    signal.wait_until_completed(OnFault::Return).unwrap();
    waiting.wait_until_completed(OnFault::Return).unwrap();
    vec![
        (
            "as the window opened: status, scheduled",
            vec![status_before, u64::from(scheduled_first)],
        ),
        (
            "the window's answers",
            answers.into_iter().map(u64::from).collect(),
        ),
        ("as they returned: the fill landed", vec![u64::from(landed)]),
        (
            "completed: status, error, the bytes",
            vec![
                waiting.status().0,
                waiting
                    .error()
                    .map_or(0, |e| u64::try_from(e.code).unwrap()),
                u64::from(read(over, 0)),
                u64::from(read(new, 0)),
            ],
        ),
    ]
}

/// What Metal does in [`behind_a_gpu_wait`] (measured on the M3 Max, macOS 27.0.1, 26A434, on a
/// GPU nothing else used: 40 runs of each case with the signal after the window and with it
/// first, 20 with it between).
///
/// Metal schedules a command buffer waiting on a plain event once a command buffer that signals
/// it was committed, the GPU then waiting, and decides its submission then. Committed before
/// any, it stays `Committed` until the signal, as behind a shared event the CPU sets
/// ([`behind_a_wait`]): a detach then answers YES at once, and the command buffer fails with
/// code 8 where it uses an object detached as the signal comes, and runs over the new pages
/// where it was replaced. Once it is scheduled, the objects it uses are in flight
/// ([`in_flight`]): a heap's detach answers NO at once and the replace after it NO, a buffer's
/// returns once the command buffer completed (104 to 125 ms, the signal's
/// kernel's time) and answers NO, YES in 2 of 120 runs, as the in-flight answer races
/// ([`expected_in_flight`]), and the command buffer runs over the pages it had. A buffer
/// detached as it was scheduled fails it with code 8 though a replace (YES) comes before the
/// signal. The calls that did not wait for a command buffer returned in 33 to 219 µs.
fn expected_behind_a_gpu_wait(order: Signal, case: &str) -> Steps {
    let detached = case != "replaced";
    let replaced = case.contains("replaced");
    let calls = usize::from(detached) + usize::from(replaced);
    let (opened, answers, landed, completed) = match order {
        Signal::After if replaced => (vec![2, 0], vec![1; calls], 0, vec![4, 0, 0x01, 0x5c]),
        Signal::After => (vec![2, 0], vec![1; calls], 0, vec![5, 8, 0x01, 0x03]),
        _ if !detached => (vec![3, 1], vec![1], 0, vec![5, 8, 0x01, 0x03]),
        _ => (
            vec![3, 1],
            vec![0; calls],
            u64::from(!case.starts_with("heap")),
            vec![4, 0, 0x5c, 0x03],
        ),
    };
    vec![
        ("as the window opened: status, scheduled", opened),
        ("the window's answers", answers),
        ("as they returned: the fill landed", vec![landed]),
        ("completed: status, error, the bytes", completed),
    ]
}

/// `steps` of [`behind_a_gpu_wait`] without the answers of a buffer's detach in flight and the
/// replace after it, which Metal does not decide ([`expected_behind_a_gpu_wait`]).
fn gpu_wait_answer_not_compared(mut steps: Steps, order: Signal, case: &str) -> Steps {
    if order != Signal::After && case.starts_with("detached") {
        let (name, words) = &mut steps[1];
        assert_eq!(*name, "the window's answers");
        *name = "the window's answers, not compared";
        words.clear();
    }
    steps
}

/// [`behind_a_gpu_wait`] on Metal, each case with the signal after the window, first and
/// between, a buffer's answers in flight not compared ([`gpu_wait_answer_not_compared`]): run
/// alone, as other GPU work (a fault's recovery) changes when Metal's command buffers complete
#[test]
fn behind_a_gpu_wait_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    for order in [Signal::After, Signal::First, Signal::Between] {
        for case in BEHIND_A_GPU_WAIT {
            let seen = behind_a_gpu_wait(&p, &dev, case, order);
            assert_eq!(
                gpu_wait_answer_not_compared(seen, order, case),
                gpu_wait_answer_not_compared(expected_behind_a_gpu_wait(order, case), order, case),
                "{case}, signal {order:?}"
            );
        }
    }
}

/// The cases of [`behind_gpu_waits`].
const BEHIND_GPU_WAITS: [&str; 5] = [
    "a lower value",
    "two events",
    "signalled before",
    "a higher value",
    "a higher value after",
];

/// Case `case` of [`BEHIND_GPU_WAITS`], over a buffer or, for `heap`, a heap over guest pages
/// ([`gpu_wait_target`]): a command buffer on one queue waits on plain events (`newEvent`) and
/// fills the buffer. For `a lower value`, it waits on an event at 2, which a command buffer on a
/// second queue signals at 1 behind [`SPIN`]'s kernel of [`ROUNDS`] rounds, committed before
/// it; a second signaller, at 2, follows on the second queue after the window. For `two events`,
/// it waits on two events at 1, one signalled so, committed before it, the other by a second
/// signaller on a third queue after the window, behind [`SPIN`]'s kernel too. For `signalled
/// before`, the event was signalled at 1 by a command buffer that completed before the waiting
/// one was committed, which runs [`SPIN`]'s kernel before its fill. For `a higher value`, it
/// waits on an event at 2, which the command buffer committed before it signals at 3, behind
/// the kernel; for `a higher value after`, that one signals another event, and a second
/// signaller, at 3, follows on the second queue after the window. With the waiting command
/// buffer committed for [`STAND`], the window: the object is detached on a thread of its own,
/// and the second signaller, where there is one, is committed once the detach returned or
/// [`BLOCKED`] passed. What shows: the waiting command buffer's status as the window opened;
/// the detach's answer, whether the fill had landed in the guest's pages as it returned and
/// whether it returned before the second signaller's commit; whether the waiting command buffer
/// was scheduled before the window and before the second signaller's commit, and the
/// signallers' statuses then (0 for none); and its status, error code and the first byte of the
/// guest's pages, once all completed.
fn behind_gpu_waits<P: Answers>(p: &P, dev: &P::Device, case: &str, heap: bool) -> Steps {
    let over = pages(if heap { 4 } else { 1 }, 0x01);
    let (target, made) = gpu_wait_target(p, dev, heap, over, true);
    let (first, second, third) = (
        dev.new_command_queue().unwrap(),
        dev.new_command_queue().unwrap(),
        dev.new_command_queue().unwrap(),
    );
    let (gate, other_gate) = (
        dev.new_event(OnFault::Return).unwrap().unwrap(),
        dev.new_event(OnFault::Return).unwrap().unwrap(),
    );
    let spun = |words: &P::Buffer| {
        // SAFETY: the buffer's own Shared contents, one page.
        unsafe { words.contents().as_ptr::<u32>().write_volatile(ROUNDS) };
    };
    let (words, waiting_words, later_words, other) = (
        plain_buffer::<P>(dev),
        plain_buffer::<P>(dev),
        plain_buffer::<P>(dev),
        plain_buffer::<P>(dev),
    );
    let signaller = command_buffer::<P>(&second);
    if case == "signalled before" {
        blit::<P>(&signaller, vec![fill::<P>(&other, 0, 0x11)]);
    } else {
        spun(&words);
        spin::<P>(dev, &signaller, &words);
    }
    let (signalled, value) = match case {
        "a higher value" => (&gate, 3),
        "a higher value after" => (&other_gate, 1),
        _ => (&gate, 1),
    };
    signaller
        .encode_signal(Some(EventRef::Event(signalled)), value)
        .unwrap();
    signaller.commit(OnFault::Return).unwrap();
    if case == "signalled before" {
        signaller.wait_until_completed(OnFault::Return).unwrap();
    }
    let waiting = command_buffer::<P>(&first);
    match case {
        "a lower value" | "a higher value" | "a higher value after" => {
            waiting
                .encode_wait(Some(EventRef::Event(&gate)), 2)
                .unwrap();
        }
        "two events" => {
            waiting
                .encode_wait(Some(EventRef::Event(&gate)), 1)
                .unwrap();
            waiting
                .encode_wait(Some(EventRef::Event(&other_gate)), 1)
                .unwrap();
        }
        _ => {
            waiting
                .encode_wait(Some(EventRef::Event(&gate)), 1)
                .unwrap();
            spun(&waiting_words);
            spin::<P>(dev, &waiting, &waiting_words);
        }
    }
    blit::<P>(&waiting, vec![fill::<P>(&target, 0, 0x5c)]);
    waiting.commit(OnFault::Return).unwrap();
    let later = match case {
        "a lower value" => Some((command_buffer::<P>(&second), &gate, 2)),
        "a higher value after" => Some((command_buffer::<P>(&second), &gate, 3)),
        "two events" => Some((command_buffer::<P>(&third), &other_gate, 1)),
        _ => None,
    };
    if let Some((later, event, value)) = &later {
        if case == "two events" {
            spun(&later_words);
            spin::<P>(dev, later, &later_words);
        } else {
            blit::<P>(later, vec![fill::<P>(&other, 256, 0x22)]);
        }
        later
            .encode_signal(Some(EventRef::Event(*event)), *value)
            .unwrap();
    }
    let clock = p.clock();
    let mut steps = std::thread::scope(|s| {
        let scheduled = s.spawn(|| {
            waiting.wait_until_scheduled().unwrap();
            let later = later.as_ref().map(|(later, _, _)| later.status().0);
            (clock.now_mono(), signaller.status().0, later)
        });
        std::thread::sleep(STAND);
        let status = waiting.status().0;
        let opened = clock.now_mono();
        let (returned, calls) = mpsc::channel();
        let (target, made) = (&target, &made);
        let window = s.spawn(move || {
            let answer = match made {
                Some(heap) => P::detach_heap(heap),
                None => P::detach_buffer(target),
            };
            let landed = read(over, 0) == 0x5c;
            returned.send(()).unwrap();
            (answer, landed)
        });
        let returned_first = calls.recv_timeout(BLOCKED).is_ok();
        let second_committed = later.as_ref().map(|(later, _, _)| {
            later.commit(OnFault::Return).unwrap();
            clock.now_mono()
        });
        let (answer, landed) = window.join().unwrap();
        let (scheduled_at, signaller_status, later_status) = scheduled.join().unwrap();
        let before = |at: MonoTime| u64::from(scheduled_at.as_nanos() < at.as_nanos());
        let steps: Steps = vec![
            ("at the window: status", vec![status]),
            (
                "the detach: answer, the fill landed, returned before the second commit",
                vec![
                    u64::from(answer),
                    u64::from(landed),
                    u64::from(returned_first),
                ],
            ),
            (
                "scheduled: before the window, before the second commit; the signallers' status",
                vec![
                    before(opened),
                    second_committed.map_or(1, before),
                    signaller_status,
                    later_status.unwrap_or(0),
                ],
            ),
        ];
        steps
    });
    if let Some((later, _, _)) = &later {
        later.wait_until_completed(OnFault::Return).unwrap();
    }
    signaller.wait_until_completed(OnFault::Return).unwrap();
    waiting.wait_until_completed(OnFault::Return).unwrap();
    steps.push((
        "completed: status, error, the byte",
        vec![
            waiting.status().0,
            waiting
                .error()
                .map_or(0, |e| u64::try_from(e.code).unwrap()),
            u64::from(read(over, 0)),
        ],
    ));
    steps
}

/// What Metal does in [`behind_gpu_waits`] (measured on the M3 Max, macOS 27.0.1, 26A434, 20 runs
/// of each case over a buffer and over a heap, on a GPU nothing else used).
///
/// Metal schedules a command buffer waiting on plain events once, for each of them, a command
/// buffer that signals it at the value waited for or past it was committed, or the event holds
/// such a value. A signaller of a lower value, or of one of two events, leaves it `Committed`,
/// and a detach then answers YES at once (30 to 440 µs). It is scheduled 197 to 507 µs after
/// the second signaller's commit, at the value or past it, while the signallers' kernels still
/// run, and then fails with code 8 over the object detached, running nothing. Behind an event a
/// completed command buffer signalled, or one a command buffer committed before it signals past
/// the value, it is scheduled at its commit, and the objects it uses are in flight: a heap's
/// detach answers NO at once (51 to 108 µs), a buffer's once the command buffer completed (104
/// to 109 ms), NO in all 40 runs, as the in-flight answer may race ([`expected_in_flight`]); it
/// then runs over the pages it had.
fn expected_behind_gpu_waits(case: &str, heap: bool) -> Steps {
    let (status, detach, scheduled, completed) = match case {
        "signalled before" | "a higher value" => (
            3,
            vec![0, u64::from(!heap), 1],
            vec![1, 1, if case == "a higher value" { 3 } else { 4 }, 0],
            vec![4, 0, 0x5c],
        ),
        _ => (2, vec![1, 0, 1], vec![0, 0, 3, 3], vec![5, 8, 0x01]),
    };
    vec![
        ("at the window: status", vec![status]),
        (
            "the detach: answer, the fill landed, returned before the second commit",
            detach,
        ),
        (
            "scheduled: before the window, before the second commit; the signallers' status",
            scheduled,
        ),
        ("completed: status, error, the byte", completed),
    ]
}

/// `steps` of [`behind_gpu_waits`] without the answer of a buffer's detach in flight, which
/// Metal does not decide ([`expected_in_flight`]).
fn gpu_waits_answer_not_compared(mut steps: Steps, case: &str, heap: bool) -> Steps {
    if matches!(case, "signalled before" | "a higher value") && !heap {
        let (name, words) = &mut steps[1];
        assert_eq!(
            *name,
            "the detach: answer, the fill landed, returned before the second commit"
        );
        *name = "the detach: the fill landed, returned before the second commit";
        words.remove(0);
    }
    steps
}

/// [`behind_gpu_waits`] on Metal, each case over a buffer and over a heap, a buffer's answer in
/// flight not compared ([`gpu_waits_answer_not_compared`]): run alone, as other GPU work (a
/// fault's recovery) changes when Metal's command buffers complete
#[test]
fn behind_gpu_waits_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    for case in BEHIND_GPU_WAITS {
        for heap in [false, true] {
            assert_eq!(
                gpu_waits_answer_not_compared(behind_gpu_waits(&p, &dev, case, heap), case, heap),
                gpu_waits_answer_not_compared(expected_behind_gpu_waits(case, heap), case, heap),
                "{case}, heap {heap}"
            );
        }
    }
}

/// Two fresh pages of the host after `case`: 0 leaves them readable and writable, 1 takes all
/// access from the second, 2 from both, 3 unmaps the second, 4 leaves both read-only.
fn pages_after(case: u8) -> HostPtr {
    let vm = MetalVm::new();
    let at = pages(2, 0x5a);
    let second = at.wrapping_add(PAGE);
    match case {
        1 => vm.protect_max(second, PAGE, VmProt(0)).unwrap(),
        2 => vm.protect_max(at, 2 * PAGE, VmProt(0)).unwrap(),
        3 => assert!(vm.deallocate(second, PAGE).is_success()),
        4 => vm.protect_max(at, 2 * PAGE, VmProt::READ).unwrap(),
        _ => {}
    }
    at
}

/// The heaps cases' options: Shared, and read-only (resource option bit 24) or not.
fn heap_options(read_only: bool) -> Vec<OptionSet> {
    let mut options = vec![OptionSet::StorageMode(0)];
    if read_only {
        options.push(OptionSet::ResourceOptionsOr(0x100_0000));
    }
    options
}

/// What a platform makes over the two pages of each of [`pages_after`]'s cases: a heap over
/// their address range, read-only or not, and a replace of a detached buffer's and a detached
/// heap's memory by them, read-only or not; 1 for each made, by case.
fn without_access<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let cases = [0, 1, 2, 3, 4];
    let vm = MetalVm::new();
    let mut steps: Steps = Vec::new();
    for read_only in [false, true] {
        let heaps = cases.map(|case| {
            let at = pages_after(case);
            let array = p
                .new_address_range_array(&[range(at, 0, 2 * PAGE)])
                .unwrap();
            let made = dev
                .new_heap(&HeapDesc {
                    size: 2 * PAGE,
                    heap_type: 1,
                    options: heap_options(read_only),
                    address_ranges: Some(&array),
                    pinned_gpu_address: None,
                })
                .unwrap();
            let made = u64::from(made.heap.is_some());
            let _ = vm.deallocate(at, 2 * PAGE);
            made
        });
        steps.push((
            if read_only {
                "read-only heaps"
            } else {
                "heaps"
            },
            heaps.to_vec(),
        ));
    }
    for read_only in [false, true] {
        let replaced = cases.map(|case| {
            let buffer = detached_buffer::<P>(dev, 2);
            let at = pages_after(case);
            let array = p
                .new_address_range_array(&[range(at, 0, 2 * PAGE)])
                .unwrap();
            let replaced = u64::from(buffer.replace_backing(&array, read_only).unwrap());
            drop(buffer);
            let _ = vm.deallocate(at, 2 * PAGE);
            replaced
        });
        steps.push((
            if read_only {
                "read-only buffer replaces"
            } else {
                "buffer replaces"
            },
            replaced.to_vec(),
        ));
    }
    for read_only in [false, true] {
        let replaced = cases.map(|case| {
            let dummy = pages(2, 0);
            let array = p
                .new_address_range_array(&[range(dummy, 0, 2 * PAGE)])
                .unwrap();
            let heap = dev
                .new_heap(&HeapDesc {
                    size: 2 * PAGE,
                    heap_type: 1,
                    options: heap_options(false),
                    address_ranges: Some(&array),
                    pinned_gpu_address: None,
                })
                .unwrap()
                .heap
                .unwrap();
            heap.detach_backing().unwrap();
            let at = pages_after(case);
            let array = p
                .new_address_range_array(&[range(at, 0, 2 * PAGE)])
                .unwrap();
            let replaced = u64::from(heap.replace_backing(&array, read_only).unwrap());
            drop(heap);
            let _ = vm.deallocate(at, 2 * PAGE);
            let _ = vm.deallocate(dummy, 2 * PAGE);
            replaced
        });
        steps.push((
            if read_only {
                "read-only heap replaces"
            } else {
                "heap replaces"
            },
            replaced.to_vec(),
        ));
    }
    steps
}

/// What the Metal platform makes over pages without access, measured on
/// `AppleM3Max-40c-25G229`: nothing over a page mapped without access or unmapped, as over
/// read-only pages what it makes over readable and writable ones.
fn expected_without_access() -> Steps {
    vec![
        ("heaps", vec![1, 0, 0, 0, 1]),
        ("read-only heaps", vec![1, 0, 0, 0, 1]),
        ("buffer replaces", vec![1, 0, 0, 0, 1]),
        ("read-only buffer replaces", vec![1, 0, 0, 0, 1]),
        ("heap replaces", vec![1, 0, 0, 0, 1]),
        ("read-only heap replaces", vec![1, 0, 0, 0, 1]),
    ]
}

/// What a platform makes as a heap of one page over ranges of two pages, read-only or not, the
/// second page past its size: one range over both of [`pages_after`]'s pages, then one range
/// of a page each, with the second readable and writable, without access, and unmapped (its
/// cases 0, 1 and 3), and a first range followed by a NULL one; 1 for each made.
fn past_the_size<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let vm = MetalVm::new();
    let mut steps: Steps = Vec::new();
    for read_only in [false, true] {
        let mut made = Vec::new();
        for split in [false, true] {
            for case in [0, 1, 3] {
                let at = pages_after(case);
                let ranges = match split {
                    false => vec![range(at, 0, 2 * PAGE)],
                    true => vec![range(at, 0, PAGE), range(at, PAGE, PAGE)],
                };
                made.push(heap_over::<P>(p, dev, &ranges, read_only));
                let _ = vm.deallocate(at, 2 * PAGE);
            }
        }
        let at = pages(1, 0x5a);
        let null = ResourceAddressRange {
            address: 0,
            length: PAGE,
        };
        made.push(heap_over::<P>(
            p,
            dev,
            &[range(at, 0, PAGE), null],
            read_only,
        ));
        let _ = vm.deallocate(at, PAGE);
        let name = match read_only {
            false => "heaps past the size",
            true => "read-only heaps past the size",
        };
        steps.push((name, made));
    }
    steps
}

/// 1 where a platform makes a Shared heap of one page over `ranges`, read-only or not, else 0.
fn heap_over<P: Platform>(
    p: &P,
    dev: &P::Device,
    ranges: &[ResourceAddressRange],
    read_only: bool,
) -> u64 {
    let array = p.new_address_range_array(ranges).unwrap();
    let made = dev
        .new_heap(&HeapDesc {
            size: PAGE,
            heap_type: 1,
            options: heap_options(read_only),
            address_ranges: Some(&array),
            pinned_gpu_address: None,
        })
        .unwrap();
    u64::from(made.heap.is_some())
}

/// What the Metal platform makes past a heap's size, measured on `AppleM3Max-40c-25G229`: as
/// within it, nothing where a page of the ranges past it is mapped without access or unmapped,
/// in one range or a range of its own, and nothing after a NULL range.
fn expected_past_the_size() -> Steps {
    vec![
        ("heaps past the size", vec![1, 0, 0, 1, 0, 0, 0]),
        ("read-only heaps past the size", vec![1, 0, 0, 1, 0, 0, 0]),
    ]
}

#[test]
fn pages_past_the_size_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    assert_eq!(past_the_size(&p, &dev), expected_past_the_size());
}

/// This crate in [`past_the_size`], in a child copy of this test binary under strict mode
/// (`REIMS_VGPU_STRICT=1`), where a death ends the process as in production: the first heap of the read-only option (resource option bit 24)
/// dies where it is made, at `newHeapWithDescriptor:`, naming `private-resource-option` (of
/// `reims_vgpu_vulkan::answer::DIES`), which the runtime announces as it makes the heap as
/// without the option, where Metal makes it and keeps the GPU's writes from the CPU
#[test]
fn pages_past_the_size_on_vulkan() {
    const CHILD: &str = "RVG_PAGES_PAST_THE_SIZE";
    if std::env::var_os(CHILD).is_some() {
        reims_vgpu_core::testkit::capture::set_capture_enabled(false);
        let dev = VulkanPlatform::system_default_device().unwrap();
        let p = VulkanPlatform::new(Box::new(NoLauncher));
        println!("made {:?}", past_the_size(&p, &dev));
        std::process::exit(0);
    }
    if VulkanPlatform::system_default_device().is_none() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "pages_past_the_size_on_vulkan", "--nocapture"])
        .args(["--test-threads=1"])
        .env(CHILD, "1")
        .env("REIMS_VGPU_STRICT", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let death = stderr
        .lines()
        .find(|l| l.starts_with("reims-vgpu: fatal: "));
    let death = death.unwrap_or_else(|| panic!("no death: {out:?}"));
    assert_eq!(out.status.signal(), Some(6), "{out:?}");
    let code = "metal2vulkan-degrade-private-resource-option";
    assert!(
        death.contains(&format!("the Vulkan runtime degraded it ({code})"))
            && death.contains("newHeapWithDescriptor:"),
        "{death}"
    );
}

#[test]
fn pages_without_access_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    assert_eq!(without_access(&p, &dev), expected_without_access());
}

/// This crate in [`without_access`], in a child copy of this test binary under strict mode
/// (`REIMS_VGPU_STRICT=1`), where a death ends the process as in production: the first heap of the read-only option (resource option bit
/// 24) dies where it is made, at `newHeapWithDescriptor:`, naming `private-resource-option` (of
/// `reims_vgpu_vulkan::answer::DIES`), which the runtime announces as it makes the heap as
/// without the option, where Metal makes it and keeps the GPU's writes from the CPU
#[test]
fn pages_without_access_on_vulkan() {
    const CHILD: &str = "RVG_PAGES_WITHOUT_ACCESS";
    if std::env::var_os(CHILD).is_some() {
        reims_vgpu_core::testkit::capture::set_capture_enabled(false);
        let dev = VulkanPlatform::system_default_device().unwrap();
        let p = VulkanPlatform::new(Box::new(NoLauncher));
        println!("made {:?}", without_access(&p, &dev));
        std::process::exit(0);
    }
    if VulkanPlatform::system_default_device().is_none() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "pages_without_access_on_vulkan", "--nocapture"])
        .args(["--test-threads=1"])
        .env(CHILD, "1")
        .env("REIMS_VGPU_STRICT", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let death = stderr
        .lines()
        .find(|l| l.starts_with("reims-vgpu: fatal: "));
    let death = death.unwrap_or_else(|| panic!("no death: {out:?}"));
    assert_eq!(out.status.signal(), Some(6), "{out:?}");
    let code = "metal2vulkan-degrade-private-resource-option";
    assert!(
        death.contains(&format!("the Vulkan runtime degraded it ({code})"))
            && death.contains("newHeapWithDescriptor:"),
        "{death}"
    );
}

/// How many shapes [`unaligned_shape`] holds.
const UNALIGNED: usize = 8;

/// The address ranges off whole pages [`unaligned`] takes, by index, over guest pages whose
/// 32-bit words hold their own offset there, top bit set, and the pages of the heap or buffer
/// over them:
/// 0. one range from 0x40 into a page, of two pages;
/// 1. one from 0x1000 into a page, of two pages;
/// 2. a range of 0x1000 bytes before one of two pages;
/// 3. a range of a page and 0x40 bytes before one of a page;
/// 4. a range of a page before one of a page from 0x40 into its page;
/// 5. the same before a third range, of a page, over three pages: whether the second takes the
///    pages its bytes lie in (pages 2 and 3, the third range past the size) or its length rounded
///    up (page 2, then the third range's page 5);
/// 6. a range of a page and 0x40 bytes before one whose bytes follow on from it, to the end of
///    page 2, over three pages: whether the two are one run of bytes (pages 0 to 2) or each takes
///    its own pages (pages 0 and 1, then 1 again);
/// 7. a range of a page, one of 0x1000 bytes from a page's start and one of a page, over three
///    pages.
///
/// A replace of shapes 2, 3 and 7, whose lengths total other than the buffer's, answers NO.
fn unaligned_shape(guest: HostPtr, index: usize) -> (Vec<ResourceAddressRange>, u64) {
    let r = |offset, length| range(guest, offset, length);
    match index {
        0 => (vec![r(0x40, 2 * PAGE)], 2),
        1 => (vec![r(0x1000, 2 * PAGE)], 2),
        2 => (vec![r(0, 0x1000), r(2 * PAGE, 2 * PAGE)], 2),
        3 => (vec![r(0, PAGE + 0x40), r(3 * PAGE, PAGE)], 2),
        4 => (vec![r(0, PAGE), r(2 * PAGE + 0x40, PAGE)], 2),
        5 => (
            vec![r(0, PAGE), r(2 * PAGE + 0x40, PAGE), r(5 * PAGE, PAGE)],
            3,
        ),
        6 => (vec![r(0, PAGE + 0x40), r(PAGE + 0x40, 2 * PAGE - 0x40)], 3),
        _ => (vec![r(0, PAGE), r(2 * PAGE, 0x1000), r(4 * PAGE, PAGE)], 3),
    }
}

/// The guest pages [`unaligned`] lies over.
const UNALIGNED_PAGES: u64 = 8;

/// What `P` does with the shape `index` ([`unaligned_shape`]) as a placement heap's address
/// ranges (`heap` true) or as a detached buffer's replace: whether it is made (or answers YES);
/// then, of a buffer of the shape's pages (the replaced one, or one placed at 0 in a heap of that
/// size), the words `contents` reads and those a GPU copy reads at the start, 0x40 and 0x1000
/// into each page and its last word, and the runs of the guest's words a GPU fill of the whole
/// buffer reached, as guest offsets.
fn unaligned<P: Platform>(p: &P, dev: &P::Device, index: usize, heap: bool) -> Steps {
    let guest = pages(UNALIGNED_PAGES, 0);
    for at in (0..UNALIGNED_PAGES * PAGE).step_by(4) {
        // SAFETY: a word of the test's own pages, which no GPU work uses yet.
        unsafe {
            guest
                .wrapping_add(at)
                .as_ptr::<u32>()
                .write(0x8000_0000 | u32::try_from(at).unwrap());
        }
    }
    let (ranges, count) = unaligned_shape(guest, index);
    let array = p.new_address_range_array(&ranges).unwrap();
    let length = count * PAGE;
    let mut steps: Steps = Vec::new();
    let (buffer, heap) = match heap {
        true => {
            let made = dev
                .new_heap(&HeapDesc {
                    size: length,
                    heap_type: 1,
                    options: vec![OptionSet::StorageMode(0)],
                    address_ranges: Some(&array),
                    pinned_gpu_address: None,
                })
                .unwrap();
            steps.push(("heap made", vec![u64::from(made.heap.is_some())]));
            let Some(h) = made.heap else {
                assert!(vm_free(guest, UNALIGNED_PAGES * PAGE));
                return steps;
            };
            let b = h.new_buffer_of_length(length, 0, 0).unwrap();
            steps.push(("placed", vec![u64::from(b.is_some())]));
            let Some(b) = b else {
                drop(h);
                assert!(vm_free(guest, UNALIGNED_PAGES * PAGE));
                return steps;
            };
            (b, Some(h))
        }
        false => {
            let b = detached_buffer::<P>(dev, count);
            let answer = b.replace_backing(&array, false).unwrap();
            steps.push(("replaced", vec![u64::from(answer)]));
            if !answer {
                drop(b);
                assert!(vm_free(guest, UNALIGNED_PAGES * PAGE));
                return steps;
            }
            (b, None)
        }
    };
    let offsets: Vec<u64> = (0..count)
        .flat_map(|page| [0, 0x40, 0x1000, PAGE - 4].map(|at| page * PAGE + at))
        .collect();
    let words = |at: HostPtr| -> Vec<u64> {
        offsets
            .iter()
            .map(|&offset| u64::from(word(at.wrapping_add(offset), 0)))
            .collect()
    };
    steps.push(("contents", words(buffer.contents())));
    let out = dev
        .new_buffer_of_length(OnFault::Return, length, 0)
        .unwrap()
        .unwrap();
    let queue = dev.new_command_queue().unwrap();
    let cb = command_buffer::<P>(&queue);
    let copy = BlitCmd::CopyBufferToBuffer {
        source_buffer: Some(&buffer),
        source_offset: 0,
        destination_buffer: Some(&out),
        destination_offset: 0,
        size: length,
    };
    blit::<P>(&cb, vec![copy]);
    steps.push(("copy", finish::<P>(&cb)));
    steps.push(("the GPU read", words(out.contents())));
    let cb = command_buffer::<P>(&queue);
    let fill = BlitCmd::FillBuffer {
        buffer: &buffer,
        range: NsRange::new(0, length),
        value: 0x5c,
    };
    blit::<P>(&cb, vec![fill]);
    steps.push(("fill", finish::<P>(&cb)));
    let mut runs: Vec<u64> = Vec::new();
    for at in (0..UNALIGNED_PAGES * PAGE).step_by(4) {
        let filled = word(guest.wrapping_add(at), 0) == 0x5c5c_5c5c;
        match (filled, runs.len() % 2) {
            (true, 0) | (false, 1) => runs.push(at),
            _ => {}
        }
    }
    if runs.len() % 2 == 1 {
        runs.push(UNALIGNED_PAGES * PAGE);
    }
    steps.push(("filled guest runs", runs));
    drop((cb, queue, out, buffer, heap));
    assert!(vm_free(guest, UNALIGNED_PAGES * PAGE));
    steps
}

/// Address ranges off whole pages ([`unaligned_shape`]), as a heap's and as a replace's, on this
/// platform as on Metal, which runs first: shapes 2 to 7 alike, step by step. A first range
/// starting off its page (shapes 0 and 1) dies at the creator here, where Metal's `contents`
/// answers its bytes from the range's own address and its GPU reads them from its page's start
/// (`src/gpu/backing.rs`'s `off_the_page`), which Metal's first words show
#[test]
fn unaligned_ranges_on_vulkan_are_metals() {
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let cases: Vec<(usize, bool)> = (0..UNALIGNED)
        .flat_map(|index| [(index, true), (index, false)])
        .collect();
    let theirs: Vec<Steps> = cases
        .iter()
        .map(|&(index, heap)| unaligned(&mp, &md, index, heap))
        .collect();
    let dev = VulkanPlatform::system_default_device().unwrap();
    let p = VulkanPlatform::new(Box::new(NoLauncher));
    for (&(index, heap), theirs) in cases.iter().zip(&theirs) {
        let (case, sel) = match heap {
            true => (format!("heap {index}"), "newHeapWithDescriptor:"),
            false => (
                format!("replace {index}"),
                "replaceBackingWithRanges:readOnly:",
            ),
        };
        if index < 2 {
            let first = |name: &str| {
                theirs
                    .iter()
                    .find(|(step, _)| *step == name)
                    .map(|(_, words)| words[0])
            };
            let start = 0x8000_0000 | [0x40, 0x1000][index];
            assert_eq!(
                (first("contents"), first("the GPU read")),
                (Some(start), Some(0x8000_0000)),
                "Metal {case}: {theirs:x?}"
            );
            let site = catch_death(|| unaligned(&p, &dev, index, heap)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{case}");
            continue;
        }
        let ours = unaligned(&p, &dev, index, heap);
        assert!(
            ours == *theirs,
            "{case}:\nours  {ours:x?}\nMetal {theirs:x?}"
        );
    }
}

/// The kernel that keeps the GPU busy (`src/gpu/spin.metal`): its first word's count of rounds,
/// about 60 µs each on the M3 Max, then [`SPUN`] in its second word.
const SPIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spin.metallib"));

/// What [`SPIN`]'s kernel writes to its second word once it ran.
const SPUN: u32 = 0x600d;

/// The rounds [`spinning`]'s kernel runs: about 120 ms on the M3 Max, past the detach and the
/// replace a test makes while it runs, each well under a millisecond, and past a late
/// scheduling. Another process's GPU fault can end the kernel's command buffer with an error as
/// its victim, on Metal as here, so the tests that run it are run again alone before a failure
/// counts.
const ROUNDS: u32 = 2000;

/// A command buffer of `queue` whose kernel spins over `words`, whose first word holds the
/// rounds it runs, committed and scheduled: submitted, and running for the rounds' time.
fn spinning<P: Platform>(
    dev: &P::Device,
    queue: &P::CommandQueue,
    words: &P::Buffer,
) -> P::CommandBuffer {
    let cb = command_buffer::<P>(queue);
    spin::<P>(dev, &cb, words);
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_scheduled().unwrap();
    cb
}

/// [`SPIN`]'s kernel over `words`, whose first word holds the rounds it runs, encoded in `cb`.
fn spin<P: Platform>(dev: &P::Device, cb: &P::CommandBuffer, words: &P::Buffer) {
    let library = dev.new_library(SPIN).unwrap().unwrap().unwrap();
    let function = library
        .new_function(OnFault::Return, "spin")
        .unwrap()
        .unwrap();
    let pipeline = dev
        .new_compute_pipeline_state_for_function(Some(&function))
        .unwrap()
        .unwrap();
    let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
    for command in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&pipeline),
        },
        ComputeCmd::SetBuffers {
            buffers: &[Some(words)],
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreadgroups {
            threadgroups_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Return, command).unwrap();
    }
}

/// Guest pages of `count` pages whose first word holds [`ROUNDS`], for [`spinning`].
fn rounds(count: u64) -> HostPtr {
    let guest = pages(count, 0);
    // SAFETY: the test's own mapped, writable page.
    unsafe { guest.as_ptr::<u32>().write_volatile(ROUNDS) };
    guest
}

/// A buffer or a heap, as core makes them (over a temporary allocation or range, detached at
/// once, replaced by the guest's pages), detached while a command buffer submitted and not yet
/// completed runs a kernel over it (over a buffer placed in the heap): whether the kernel had
/// ended as the detach returned, and the answer of a replace after it; the command buffer's
/// status, error code and whether the kernel's write landed in the guest's pages, once it
/// completed; then, once Metal retired its use ([`retire`]), the answer of a replace after a
/// second detach.
fn in_flight<P: Platform>(p: &P, dev: &P::Device, heap: bool) -> Steps {
    let queue = dev.new_command_queue().unwrap();
    let size = 4 * PAGE;
    let guest = rounds(4);
    let array = p.new_address_range_array(&[range(guest, 0, size)]).unwrap();
    let (words, made) = if heap {
        let dummy = pages(4, 0);
        let ranges = p.new_address_range_array(&[range(dummy, 0, size)]).unwrap();
        let made = dev
            .new_heap(&HeapDesc {
                size,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap();
        let options = made.descriptor_resource_options;
        let heap = made.heap.unwrap();
        let words = heap
            .new_buffer_of_length(PAGE, options, 0)
            .unwrap()
            .unwrap();
        heap.detach_backing().unwrap();
        assert!(heap.replace_backing(&array, false).unwrap());
        assert!(vm_free(dummy, size));
        (words, Some(heap))
    } else {
        let buffer = detached_buffer::<P>(dev, 4);
        assert!(buffer.replace_backing(&array, false).unwrap());
        (buffer, None)
    };
    let detach = || match &made {
        Some(heap) => heap.detach_backing().unwrap(),
        None => words.detach_backing().unwrap(),
    };
    let fresh = pages(4, 0);
    let array = p.new_address_range_array(&[range(fresh, 0, size)]).unwrap();
    let replace = || match &made {
        Some(heap) => heap.replace_backing(&array, false).unwrap(),
        None => words.replace_backing(&array, false).unwrap(),
    };
    let cb = spinning::<P>(dev, &queue, &words);
    detach();
    let ended = word(guest, 1) == SPUN;
    let mut steps: Steps = vec![(
        "detached in flight: ended, replaced",
        vec![u64::from(ended), u64::from(replace())],
    )];
    cb.wait_until_completed(OnFault::Return).unwrap();
    steps.push((
        "completed: status, error, the kernel's write",
        vec![
            cb.status().0,
            cb.error().map_or(0, |e| u64::try_from(e.code).unwrap()),
            u64::from(word(guest, 1) == SPUN),
        ],
    ));
    retire();
    detach();
    steps.push((
        "detached once it completed: replaced",
        vec![u64::from(replace())],
    ));
    drop((cb, queue, words, made));
    assert!(vm_free(guest, size) && vm_free(fresh, size));
    steps
}

/// What Metal does in [`in_flight`] (measured on the M3 Max, macOS 26.7, 25G229, 28 runs of each
/// on a GPU nothing else used): a heap's detach answers NO at once, the kernel still running, and
/// a buffer's returns once the command buffer completed, the kernel's write landed, and answers
/// NO (in most runs, below); so the object stays attached, the replace after it answers NO, and
/// the command buffer completes over the guest's pages. Once it completed and its use retired, a
/// detach detaches. The status a buffer's command buffer reads as the detach returns is
/// `Completed` or, in some runs, still `Scheduled`, so it is not compared.
///
/// A buffer's answer is not decided by the state the test drives: Metal answers NO in most runs
/// and YES in some, the object then detached and the replace after it answering YES. Its detach
/// is one call into the kernel driver (`IOGPUResourceDetachBacking`, a single
/// `IOConnectCallMethod`), which waits until every submitted GPU use of the buffer completed,
/// then answers NO where the kernel has not yet retired that use and YES where it has, the
/// kernel's own completion racing the waiting call. Measured on 2026-10-11, `detachBacking`'s own
/// answer, driven as here: YES in 13 of 150 runs and in 3 of 40. Nothing the caller does decides
/// it: a completed handler held blocked until after the replace (YES in 8 of 60), more GPU work
/// after the kernel in its command buffer (9 of 80), the command buffer held in flight past the
/// detach by an event wait after the kernel (8 of 150, and 1 of 150 where the work behind the
/// wait uses the buffer too), and a second queue's command buffer using the buffer (5 of 60)
/// each left both answers. [`detached_in_flight_on_metal`] compares the rest of the buffer's
/// steps and not that answer ([`answer_not_compared`]). This crate's NO, with the replace's NO
/// after it, is one of Metal's two answers, and [`detached_in_flight_on_vulkan`] compares it.
fn expected_in_flight() -> [Steps; 2] {
    let completed = (
        "completed: status, error, the kernel's write",
        vec![4, 0, 1],
    );
    let after = ("detached once it completed: replaced", vec![1]);
    [
        vec![
            ("detached in flight: ended, replaced", vec![0, 0]),
            completed.clone(),
            after.clone(),
        ],
        vec![
            ("detached in flight: ended, replaced", vec![1, 0]),
            completed,
            after,
        ],
    ]
}

/// `steps` of [`in_flight`] without the answer of the replace after the detach in flight, which
/// Metal does not decide for a buffer ([`expected_in_flight`]): whether the kernel had ended as
/// the detach returned stays.
fn answer_not_compared(mut steps: Steps) -> Steps {
    let (name, words) = &mut steps[0];
    assert_eq!(*name, "detached in flight: ended, replaced");
    *name = "detached in flight: ended";
    words.truncate(1);
    steps
}

/// [`in_flight`] on Metal, a heap and a buffer, the buffer's answer in flight not compared
/// ([`answer_not_compared`]): run alone, as other GPU work (a fault's recovery) changes when
/// Metal's command buffers complete
#[test]
fn detached_in_flight_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    let [heap, buffer] = [true, false].map(|heap| in_flight(&p, &dev, heap));
    let [heap_expected, buffer_expected] = expected_in_flight();
    assert_eq!(heap, heap_expected, "heap");
    assert_eq!(
        answer_not_compared(buffer),
        answer_not_compared(buffer_expected),
        "buffer"
    );
}

/// [`in_flight`] on this crate, as on Metal ([`expected_in_flight`]): a detach answers NO while a
/// command buffer the runtime submitted took the object, a buffer's once each such command
/// buffer completed (`src/gpu/backing.rs`'s `in_flight`)
#[test]
fn detached_in_flight_on_vulkan() {
    let Some(dev) = VulkanPlatform::system_default_device() else {
        return;
    };
    let p = VulkanPlatform::new(Box::new(NoLauncher));
    let seen = [true, false].map(|heap| in_flight(&p, &dev, heap));
    assert_eq!(seen, expected_in_flight());
}

/// A buffer, or a buffer placed in a heap, as core makes them, over the guest's pages, used by a
/// command buffer that completed, then used by one not yet committed (its encoder ended, or still
/// open where `open`), and detached: what a kernel that reaches it through an address it loads
/// does ([`reach`]), and then the holder's status and error code, committed
/// ([`expected_held`]).
fn held<P: Platform>(p: &P, dev: &P::Device, heap: bool, open: bool) -> Steps {
    let queue = dev.new_command_queue().unwrap();
    let size = if heap { 4 * PAGE } else { PAGE };
    let guest = pages(size / PAGE, 0xa1);
    let array = p.new_address_range_array(&[range(guest, 0, size)]).unwrap();
    let (target, made) = if heap {
        let dummy = pages(4, 0);
        let ranges = p.new_address_range_array(&[range(dummy, 0, size)]).unwrap();
        let made = dev
            .new_heap(&HeapDesc {
                size,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap();
        let options = made.descriptor_resource_options;
        let heap = made.heap.unwrap();
        let target = heap
            .new_buffer_of_length(PAGE, options, 0)
            .unwrap()
            .unwrap();
        heap.detach_backing().unwrap();
        assert!(heap.replace_backing(&array, false).unwrap());
        assert!(vm_free(dummy, size));
        (target, Some(heap))
    } else {
        let buffer = detached_buffer::<P>(dev, 1);
        assert!(buffer.replace_backing(&array, false).unwrap());
        (buffer, None)
    };
    let used = command_buffer::<P>(&queue);
    blit::<P>(&used, vec![fill::<P>(&target, 256, 0x77)]);
    assert_eq!(finish::<P>(&used), [4, 0]);
    retire();
    let holder = command_buffer::<P>(&queue);
    let e = holder.blit_encoder(OnFault::Return).unwrap().unwrap();
    e.encode(OnFault::Return, fill::<P>(&target, 512, 0x66))
        .unwrap();
    if !open {
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    }
    match &made {
        Some(heap) => heap.detach_backing().unwrap(),
        None => target.detach_backing().unwrap(),
    }
    let mut steps: Steps = vec![(
        "reached unnamed after the detach",
        reach::<P>(dev, &queue, &target, guest),
    )];
    if open {
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    }
    steps.push(("the holder, committed", finish::<P>(&holder)));
    steps
}

/// What Metal does in [`held`], a buffer and a heap, the holder's encoder ended or open alike
/// (measured on the M3 Max, macOS 26.7, 25G229): the detach detaches, a kernel that reaches the
/// object unnamed reads 0 and its write is dropped, its command buffer completing with no error,
/// and the holder fails with code 8 as it is submitted.
fn expected_held() -> Steps {
    vec![
        ("reached unnamed after the detach", vec![4, 0, 0, 0]),
        ("the holder, committed", vec![5, 8]),
    ]
}

/// The cases of [`held`]: a heap or a buffer, the holder's encoder open or ended.
const HELD: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

#[test]
fn detached_while_held_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    for (heap, open) in HELD {
        assert_eq!(
            held(&p, &dev, heap, open),
            expected_held(),
            "heap {heap}, open {open}"
        );
    }
}

/// [`held`] on this crate, as on Metal ([`expected_held`]): the runtime's detach detaches the
/// object while a command buffer not yet committed used it, which fails as it is submitted, and
/// leaves it in no memory, which an unnamed reach reads as 0 (`src/gpu/backing.rs`'s docs)
#[test]
fn detached_while_held_on_vulkan() {
    let Some(dev) = VulkanPlatform::system_default_device() else {
        return;
    };
    let p = VulkanPlatform::new(Box::new(NoLauncher));
    for (heap, open) in HELD {
        assert_eq!(
            held(&p, &dev, heap, open),
            expected_held(),
            "heap {heap}, open {open}"
        );
    }
}

/// A buffer, or a buffer placed in a heap, made as core makes them over a temporary allocation
/// or range of its own whose words are 0xd0d0d0d0, detached at once and never replaced: a command
/// buffer that names it, then a kernel that reaches it through an address it loads ([`reach`]),
/// and one that reads the word that kernel wrote, both naming nothing of it; whether `contents`
/// and the temporary memory's words stayed; and, for the buffer, whether its deallocator ran
/// before its release and once it went.
fn never_replaced<P: Platform>(p: &P, dev: &P::Device, heap: bool) -> Steps {
    let queue = dev.new_command_queue().unwrap();
    let size = 4 * PAGE;
    let dummy = pages(4, 0xd0);
    let (ran, runs) = mpsc::channel();
    let (target, made, contents) = if heap {
        let ranges = p.new_address_range_array(&[range(dummy, 0, size)]).unwrap();
        let made = dev
            .new_heap(&HeapDesc {
                size,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap();
        let options = made.descriptor_resource_options;
        let heap = made.heap.unwrap();
        let target = heap
            .new_buffer_of_length(PAGE, options, 0)
            .unwrap()
            .unwrap();
        let contents = target.contents();
        heap.detach_backing().unwrap();
        (target, Some(heap), contents)
    } else {
        // SAFETY: the dummy's pages stay mapped until the deallocator, which frees them, runs.
        let contents = unsafe { HostRange::new(dummy, size) };
        let buffer = dev
            .new_buffer(BufferDesc {
                length: size,
                options: vec![OptionSet::StorageMode(0)],
                pinned_gpu_address: None,
                contents: Some(contents),
                no_copy: true,
                deallocator: Some(Box::new(move |at, length| {
                    let _ = ran.send(at == dummy);
                    let _ = MetalVm::new().deallocate(at, length);
                })),
            })
            .unwrap()
            .unwrap();
        let contents = buffer.contents();
        buffer.detach_backing().unwrap();
        (buffer, None, contents)
    };
    let named = command_buffer::<P>(&queue);
    blit::<P>(&named, vec![fill::<P>(&target, 256, 0x77)]);
    let mut steps: Steps = vec![("named", finish::<P>(&named))];
    steps.push(("reached", reach::<P>(dev, &queue, &target, dummy)));
    steps.push((
        "the written word read",
        reach_at::<P>(dev, &queue, &target, 4, dummy.wrapping_add(4)),
    ));
    steps.push((
        "contents and the temporary memory stay",
        vec![
            u64::from(target.contents() == contents),
            u64::from((0..3).all(|at| word(dummy, at) == 0xd0d0_d0d0)),
        ],
    ));
    if made.is_none() {
        steps.push((
            "the deallocator ran before the release",
            vec![u64::try_from(runs.try_iter().count()).unwrap()],
        ));
        drop(target);
        let ran = runs
            .recv_timeout(std::time::Duration::from_secs(2))
            .is_ok_and(|over_dummy| over_dummy);
        steps.push(("the deallocator ran once it went", vec![u64::from(ran)]));
    } else {
        drop((target, made));
        assert!(vm_free(dummy, size));
    }
    steps
}

/// What Metal does in [`never_replaced`], a buffer and a heap (measured on the M3 Max, macOS
/// 26.7, 25G229): the command buffer that names the object fails with code 8; a kernel that
/// reaches it unnamed reads 0 and its write is dropped, its command buffer completing with no
/// error, and a later one reads 0 where it wrote; `contents` stays, and the temporary memory
/// keeps its words; and the buffer's deallocator runs once it went, not before.
fn expected_never_replaced(heap: bool) -> Steps {
    let mut steps = vec![
        ("named", vec![5, 8]),
        ("reached", vec![4, 0, 0, 0]),
        ("the written word read", vec![4, 0, 0, 0]),
        ("contents and the temporary memory stay", vec![1, 1]),
    ];
    if !heap {
        steps.push(("the deallocator ran before the release", vec![0]));
        steps.push(("the deallocator ran once it went", vec![1]));
    }
    steps
}

#[test]
fn never_replaced_on_metal() {
    let Some(dev) = MetalPlatform::system_default_device() else {
        return;
    };
    let p = MetalPlatform::new(Box::new(NoLauncher));
    for heap in [false, true] {
        assert_eq!(
            never_replaced(&p, &dev, heap),
            expected_never_replaced(heap),
            "heap {heap}"
        );
    }
}

/// [`never_replaced`] on this crate, as on Metal: the runtime's detach leaves the object in no
/// memory, and the buffer's deallocator is held until its release (`src/gpu/backing.rs`'s
/// `Held`)
#[test]
fn never_replaced_on_vulkan() {
    let Some(dev) = VulkanPlatform::system_default_device() else {
        return;
    };
    let p = VulkanPlatform::new(Box::new(NoLauncher));
    for heap in [false, true] {
        assert_eq!(
            never_replaced(&p, &dev, heap),
            expected_never_replaced(heap),
            "heap {heap}"
        );
    }
}
