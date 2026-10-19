//! One task's connection in this process: the host's end ([`Link`], a `ServiceLink`), the
//! service's end ([`ToHostLink`], a `HostLink`), the two queues between them and the host's
//! pending replies.
//!
//! - **The connection queue** delivers the host's messages to the service one at a time, in send
//!   order, each as the task's code (`TaskLife::run`); the handshake runs there. A message sent
//!   with `send_here` is delivered in the same order on the sender's thread, which holds the
//!   queue while the service takes it. The task's end runs there too: the service's state is
//!   released, then the host is told.
//! - **The link's queue** makes every `HostSide` call and runs every reply, one at a time, in the
//!   order the service sent them. A message or reply the service hands over while the task runs
//!   is always delivered, before the end the host is told of. The reply to a message sent with
//!   `send_here`, made on the sender's thread while the queue is idle with nothing queued, runs
//!   there, holding the queue, in the same place in its order.
//! - **Pending replies**: the service gets a relay in each reply's place, so the host's `Reply`
//!   is released exactly when `ServiceLink::send` says: after its one run, right after the
//!   interruption or invalidation, or at once for a send the link drops.
//! - **Lent surfaces**: an IOSurface the host lends the service reaches it as another object over
//!   the same surface, as it reaches a process it is sent to ([`own_surface`]).

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::thread::ThreadId;

use parking_lot::{Mutex, MutexGuard};

use super::life::TaskLife;
use crate::platform::Platform;
use crate::platform::os::surface::Surface;
use crate::rt::queue::SerialQueue;
use crate::tasks::launcher::{HostLink, HostSide, ServiceLink, ServiceSide};
use crate::tasks::wire::{Reply, ToHost, ToService};

/// How a connection ends, as the host is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// The task died: the host's `on_interrupted`.
    Died,
    /// The host invalidated the connection: its `on_invalidated`.
    Invalidated,
}

/// What the link's queue runs.
enum HostEvent<P: Platform> {
    Message(ToHost<P>),
    /// A service reply: runs the host's `Reply` once and releases it.
    Reply(Box<dyn FnOnce() + Send>),
    End(End),
}

/// The connection's state, under one lock: every hand-over to the link's queue happens under it,
/// so the end the host is told of comes after every message the service handed over before it.
struct State<P: Platform> {
    /// The host side, held until the invalidation has been delivered.
    host: Option<Arc<dyn HostSide<P>>>,
    /// The service, held until the task ends.
    service: Option<Arc<dyn ServiceSide<P>>>,
    /// The host has been told, or is about to be, that the connection ended: sends are dropped.
    ended: bool,
    /// The invalidation has begun.
    invalidated: bool,
}

/// One pending host `Reply<T>`, boxed as `Any`, with its table id.
type PendingReply = (u64, Box<dyn Any + Send>);

/// One task's connection.
pub(super) struct Connection<P: Platform> {
    state: Mutex<State<P>>,
    life: Arc<TaskLife>,
    next_reply: AtomicU64,
    /// The host's `Reply`s the service has not answered; `None` once the connection is over.
    pending: Mutex<Option<Vec<PendingReply>>>,
    connection_queue: SerialQueue,
    link_queue: SerialQueue,
}

impl<P: Platform> Connection<P> {
    /// A connection to `host` for the task whose life is `life`, with no service yet.
    pub(super) fn new(
        platform: &Arc<P>,
        host: Arc<dyn HostSide<P>>,
        life: Arc<TaskLife>,
    ) -> Arc<Connection<P>> {
        let conn = Arc::new(Connection {
            state: Mutex::new(State {
                host: Some(host),
                service: None,
                ended: false,
                invalidated: false,
            }),
            life,
            next_reply: AtomicU64::new(0),
            pending: Mutex::new(Some(Vec::new())),
            connection_queue: SerialQueue::new(platform.threads(), "reims-vgpu.task-connection"),
            link_queue: SerialQueue::new(platform.threads(), "reims-vgpu.task-link"),
        });
        // The task's death ends the connection from its queue, after the dying thread's unwind.
        let weak: Weak<Connection<P>> = Arc::downgrade(&conn);
        conn.life.on_death(move || {
            if let Some(conn) = weak.upgrade() {
                conn.end_later(End::Died);
            }
        });
        conn
    }

    /// Gives the connection its service.
    pub(super) fn serve(&self, service: Arc<dyn ServiceSide<P>>) {
        self.state.lock().service = Some(service);
    }

    /// Queues the connection's end behind the messages already queued for the service.
    fn end_later(self: &Arc<Self>, end: End) {
        let conn = Arc::clone(self);
        self.connection_queue.exec_async(move || conn.end(end));
    }

    /// The connection's end, on the connection queue: the service releases its state (as the
    /// task's code: a death there stops only the release), then the host is told, after
    /// everything the service handed over before.
    fn end(self: &Arc<Self>, end: End) {
        let service = self.state.lock().service.take();
        if let Some(service) = service {
            let _ = self.life.releasing(move || {
                service.end();
                drop(service);
            });
        }
        let mut state = self.state.lock();
        // A death after the invalidation began is not the host's to hear of.
        if end == End::Died && state.ended {
            return;
        }
        state.ended = true;
        self.to_host(HostEvent::End(end));
    }

    /// Hands `event` to the link's queue. Called under the state lock.
    fn to_host(self: &Arc<Self>, event: HostEvent<P>) {
        let conn = Arc::clone(self);
        self.link_queue
            .exec_async(move || conn.on_link_queue(event));
    }

    /// One event, on the link's queue.
    fn on_link_queue(&self, event: HostEvent<P>) {
        let host = self.state.lock().host.clone();
        match event {
            HostEvent::Message(m) => {
                if let Some(host) = &host {
                    host.on_message(m);
                }
            }
            // Runs once; the closure, and with it the `Reply`, drops right after.
            HostEvent::Reply(f) => f(),
            HostEvent::End(End::Died) => {
                if let Some(host) = &host {
                    host.on_interrupted();
                }
                drop(host);
                self.release_pending();
            }
            HostEvent::End(End::Invalidated) => {
                if let Some(host) = &host {
                    host.on_invalidated();
                }
                self.release_pending();
                // The connection is over: it lets go of the host.
                let last = self.state.lock().host.take();
                drop((host, last));
            }
        }
    }

    /// One host message to the service, on the connection queue, as the task's code: on the
    /// queue's worker, or on the sender's thread holding the queue for one sent with `send_here`
    /// (`here`, [`ServiceSide::on_message_here`]). A message for a task that has ended is
    /// dropped: its relay goes, and the host's `Reply` waits in the table for the end.
    fn deliver(&self, m: ToService<P>, here: bool) {
        let service = self.state.lock().service.clone();
        let Some(service) = service else {
            return;
        };
        let _ = self.life.run(move || {
            if here {
                service.on_message_here(m)
            } else {
                service.on_message(m)
            }
        });
    }

    /// Takes the host's pending `Reply` `id` out of the table.
    fn take_pending(&self, id: u64) -> Option<Box<dyn Any + Send>> {
        let mut table = self.pending.lock();
        let t = table.as_mut()?;
        let at = t.iter().position(|(k, _)| *k == id)?;
        Some(t.remove(at).1)
    }

    /// Releases every pending `Reply` without running it, outside the table's lock (a release
    /// may run the remote task's drop), and closes the table.
    fn release_pending(&self) {
        let released = self.pending.lock().take();
        drop(released);
    }

    /// Puts a relay in `reply`'s place and keeps the host's `Reply` in the table. The relay's
    /// first call, while the task runs, hands the `Reply` to the link's queue to run once; later
    /// calls find nothing. `false` when the table is closed (the `Reply` is then released).
    ///
    /// The reply to a message sent with `send_here` from thread `here` runs on that thread
    /// instead, when the service replies there and the link's queue is idle with nothing queued
    /// ([`replies_here`]): the queue's turn, taken under the state lock, under which every
    /// hand-over to the queue happens, is the reply's place in the queue's order, and it runs
    /// holding the queue once the lock is released, outside the task's code, as the queue's
    /// worker would have run it.
    fn relay<T: Send + 'static>(
        self: &Arc<Self>,
        reply: &mut Reply<T>,
        here: Option<ThreadId>,
    ) -> bool {
        let id = self.next_reply.fetch_add(1, Ordering::Relaxed);
        let weak = Arc::downgrade(self);
        let original = std::mem::replace(
            reply,
            Reply::new(move |v: T| {
                let Some(conn) = weak.upgrade() else {
                    return;
                };
                // A task that has ended sends nothing: the host's `Reply` stays in the table
                // until the end releases it, unrun.
                let state = conn.state.lock();
                if state.ended || conn.life.has_ended() {
                    return;
                }
                let Some(r) = conn
                    .take_pending(id)
                    .and_then(|b| b.downcast::<Reply<T>>().ok())
                else {
                    return;
                };
                if here.is_some_and(|t| t == std::thread::current().id())
                    && replies_here()
                    && let Some(turn) = conn.link_queue.hold_if_idle()
                {
                    drop(state);
                    // The host's side, as on the link queue's worker: outside the task's code.
                    super::life::outside_the_task(|| turn.run(move || r.call(v)));
                    return;
                }
                conn.to_host(HostEvent::Reply(Box::new(move || r.call(v))));
                drop(state);
            }),
        );
        let mut table = self.pending.lock();
        match table.as_mut() {
            Some(t) => {
                t.push((id, Box::new(original)));
                true
            }
            None => {
                drop(table);
                drop(original);
                false
            }
        }
    }

    /// Relays the reply of `m`, if it has one; `here` is the sender's thread for a message sent
    /// with `send_here`.
    fn relay_message(self: &Arc<Self>, m: &mut ToService<P>, here: Option<ThreadId>) {
        match m {
            ToService::Handshake { reply, .. } => {
                self.relay(reply, here);
            }
            ToService::ExecIndirect3 { reply, .. } => {
                self.relay(reply, here);
            }
            ToService::FlushResources { reply }
            | ToService::UnmapMemory { reply, .. }
            | ToService::SetObjectList { reply, .. }
            | ToService::GetComputeInfo { reply, .. }
            | ToService::ReplacePhysical { reply, .. }
            | ToService::InvalidateResources { reply, .. }
            | ToService::SynchronizeResources { reply, .. }
            | ToService::DiscardResources { reply, .. }
            | ToService::HeapTextureSizeAndAlign { reply, .. }
            | ToService::DeleteObject { reply, .. }
            | ToService::DeleteResource { reply, .. }
            | ToService::InfoIndirect { reply, .. }
            | ToService::ResetRasterizationRateMap { reply, .. }
            | ToService::FifoDeleted { reply, .. } => {
                self.relay(reply, here);
            }
            ToService::SetFifoEvent { .. } => {}
        }
    }
}

/// Whether the reply to a message sent with `send_here` runs on the sender's thread when the
/// service replies there with the link's queue idle ([`Connection::relay`]): on, unless
/// `REIMS_VGPU_TASK_INLINE_REPLY` is `0`, read once per process, when the link queue's worker
/// runs it, as every other reply. The FIFO thread then finds its exec's reply run when the job
/// returns, and neither the link queue's worker nor the FIFO thread is woken for it.
fn replies_here() -> bool {
    static HERE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !std::env::var_os("REIMS_VGPU_TASK_INLINE_REPLY").is_some_and(|v| v == "0")
    });
    #[cfg(test)]
    if let Some(here) = REPLIES_HERE.with(std::cell::Cell::get) {
        return here;
    }
    *HERE
}

#[cfg(test)]
thread_local! {
    /// Whether replies on this thread run here, in place of the process's switch
    /// ([`replies_here`]).
    pub(crate) static REPLIES_HERE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// The host's end of one task's connection.
pub(super) struct Link<P: Platform>(pub(super) Arc<Connection<P>>);

impl<P: Platform> Link<P> {
    /// `m` with its reply relayed, under the state lock it answers with, or `None` once the
    /// connection has ended: `m` is then dropped, its `Reply` released at once, unrun, and the
    /// first send after the task's death also invalidates the connection. `here` is the sender's
    /// thread for a message sent with `send_here`.
    fn admit(
        &self,
        mut m: ToService<P>,
        here: Option<ThreadId>,
    ) -> Option<(ToService<P>, MutexGuard<'_, State<P>>)> {
        let conn = &self.0;
        let state = conn.state.lock();
        if state.ended {
            let invalidate = !state.invalidated;
            drop(state);
            drop(m);
            if invalidate {
                self.invalidate();
            }
            return None;
        }
        conn.relay_message(&mut m, here);
        Some((m, state))
    }
}

impl<P: Platform> ServiceLink<P> for Link<P> {
    fn send(&self, m: ToService<P>) {
        let Some((m, state)) = self.admit(m, None) else {
            return;
        };
        let c = Arc::clone(&self.0);
        self.0
            .connection_queue
            .exec_async(move || c.deliver(m, false));
        drop(state);
    }

    /// The connection queue's `sync`: the delivery runs on this thread once every message
    /// queued before it has been delivered, holding the queue until the service returns. The
    /// queue is taken after the state lock is released, since the service's own sends take that
    /// lock. An end queued in between runs first: `m` then finds no service, and its `Reply`,
    /// already in the table, goes with the end's release, unrun.
    fn send_here(&self, m: ToService<P>) {
        let Some((m, state)) = self.admit(m, Some(std::thread::current().id())) else {
            return;
        };
        drop(state);
        let c = Arc::clone(&self.0);
        self.0.connection_queue.sync(move || c.deliver(m, true));
    }

    fn invalidate(&self) {
        let conn = &self.0;
        {
            let mut state = conn.state.lock();
            if state.invalidated {
                return;
            }
            // From here a send is dropped and its `Reply` released at once.
            state.invalidated = true;
            state.ended = true;
        }
        // The task's threads stop at their next step, and one waiting on a round trip is woken.
        conn.life.exit();
        conn.end_later(End::Invalidated);
        // The service's state is gone before the host goes on: the service shares the host's
        // Metal device, so the resource IDs and GPU addresses its release frees are free before
        // the host's next allocation, in the same order every run. The wait needs nothing the
        // host may hold. The release takes no lock of the host's and sends it nothing; the
        // task's threads it waits for stop at their next step, and a round trip in flight is
        // woken by the end instead of answered; the host drains every token before it
        // invalidates, so no GPU work of the task is left to wait for. The link's queue, which
        // runs the host's side, never waits here.
        if !conn.link_queue.is_current() && !conn.connection_queue.is_current() {
            conn.connection_queue.sync(|| ());
        }
    }
}

/// The service's end of the connection.
pub(super) struct ToHostLink<P: Platform>(pub(super) Arc<Connection<P>>);

impl<P: Platform> HostLink<P> for ToHostLink<P> {
    fn send(&self, m: ToHost<P>) {
        let conn = &self.0;
        let state = conn.state.lock();
        if state.ended || conn.life.has_ended() {
            drop(state);
            drop(m);
            conn.life.stop_if_ended();
            return;
        }
        conn.to_host(HostEvent::Message(own_surface(m)));
        drop(state);
    }
}

/// `m`, with the reply of a surface retain handing the service its own object over the surface
/// the host answers with. What the service holds (its cache, the textures and buffers over the
/// surface) then never holds the host's object, whose retain count decides whether the guest may
/// unmap it. The host's reference goes when its answer returns.
fn own_surface<P: Platform>(m: ToHost<P>) -> ToHost<P> {
    match m {
        ToHost::RetainIoSurface { mapping_id, reply } => ToHost::RetainIoSurface {
            mapping_id,
            reply: Reply::new(move |surface: Option<P::Surface>| {
                reply.call(surface.and_then(|s| s.lookup_by_port()));
            }),
        },
        m => m,
    }
}
