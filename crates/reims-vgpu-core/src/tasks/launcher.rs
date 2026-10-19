//! The transport of isolated tasks: the traits between a remote task and its service.
//!
//! The host side launches one service per task ([`TaskLauncher`]) and talks to it over the link
//! it gets back ([`ServiceLink`]); the service's messages come back to the task's endpoint
//! ([`HostSide`]). The service ([`ServiceSide`], the GPU task) answers over its own link
//! ([`HostLink`]). The transport sits below the remote task and its completions, so status 4 is
//! one piece of code for every launcher. The product's launcher is
//! [`TaskThreads`](crate::tasks::isolation::TaskThreads): each task's service on threads of its
//! own in this process; the testkit wraps it to record what it launched.
//!
//! ```text
//!   RemoteTask --ToService--> dyn ServiceLink --> (the task's connection queue) dyn ServiceSide
//!   Endpoint <--ToHost-- (on the link's queue) <-- dyn HostLink <-- GpuTask
//!        ^ dyn HostSide: on_message / on_interrupted / on_invalidated
//! ```

use std::sync::Arc;

use crate::platform::Platform;
use crate::tasks::wire::{ToHost, ToService};

/// Starts one isolated task's service.
pub trait TaskLauncher<P: Platform>: Send + Sync {
    /// A new service per call, never pooled or reused, over `platform`, with `host` on its
    /// connection: the task's death (its interruption) and the connection's invalidation reach
    /// `host` as a crash.
    ///
    /// The launcher holds `host` for the connection's life. The caller sends the handshake as the
    /// first `ToService` message and waits for its reply.
    fn launch(&self, platform: &Arc<P>, host: Arc<dyn HostSide<P>>) -> Arc<dyn ServiceLink<P>>;
}

/// The host's end of one task's connection.
pub trait ServiceLink<P: Platform>: Send + Sync {
    /// One message to the service. Ordered and asynchronous: sends are delivered in call order;
    /// replies come back through the message's `Reply` on the link's queue. A send on an
    /// invalidated or interrupted link is dropped; its reply never runs, which the completions
    /// cover (their tokens were signalled 4 by the interruption). The first send after an
    /// interruption also invalidates the link, whose `on_invalidated` then runs.
    ///
    /// **When the link releases the message's `Reply`** (this decides when the remote task
    /// drops: a reply holds the task, which owns the link, and only this release breaks that
    /// cycle). The link holds the host's `Reply` from `send` until one of:
    /// - the service's first reply: the link runs the `Reply` once, on the link's queue in
    ///   order with `HostSide::on_message`, and releases it as soon as that call returns;
    /// - the connection's interruption or invalidation: the link releases every `Reply` still
    ///   pending, without running it, on the link's queue right after `HostSide::on_interrupted`
    ///   or `on_invalidated` has run and before the link's queue runs anything else. Releasing
    ///   after the handler, not before it, means the crash has signalled every token before a
    ///   released `Reply` can run the task's drop, whose drain would otherwise wait on this
    ///   same queue;
    /// - a send the link drops (above): released at once, never run.
    ///
    /// A second reply from the service never reaches the `Reply`: it is already released. The
    /// service side's `Reply` may still be called any number of times.
    fn send(&self, m: ToService<P>);

    /// [`ServiceLink::send`], delivered on the caller's thread: once every message sent before
    /// it has been delivered, the service takes it there ([`ServiceSide::on_message_here`]), and
    /// this returns when the service has. Its `Reply` is held and run as `send`'s is, in order
    /// with the service's other messages on the link's queue: on the caller's thread, holding
    /// that queue, when the service replies there with the queue idle and nothing queued. By
    /// default, `send`.
    fn send_here(&self, m: ToService<P>) {
        self.send(m)
    }

    /// Invalidates the connection: the service ends and its state is released, before this
    /// returns unless it runs on the link's queue. Idempotent. The host side's `on_invalidated`
    /// then runs on the link's queue. The wait needs no lock the caller may hold.
    fn invalidate(&self);
}

/// The host side of a connection: the device protocol's messages plus the connection's
/// lifecycle. Every call runs on the link's queue, one at a time, in the order the service sent.
pub trait HostSide<P: Platform>: Send + Sync {
    /// One message from the service.
    fn on_message(&self, m: ToHost<P>);

    /// The connection's interruption (the task died): a crash, status 4 for every token in
    /// ascending id order.
    fn on_interrupted(&self);

    /// The connection's invalidation: a crash as well.
    fn on_invalidated(&self);
}

/// The service's end of the connection back to its host.
pub trait HostLink<P: Platform>: Send + Sync {
    /// One message to the host, ordered with the replies to `ToService` messages on the same
    /// connection. Once the task has ended the message goes nowhere, and a thread running the
    /// task's code stops here.
    fn send(&self, m: ToHost<P>);
}

/// The service side of a connection (`gpu_task::GpuTask`). The transport calls it on the
/// connection's queue in arrival order; the handshake runs synchronously there, every other
/// message queues its work on the service queue.
pub trait ServiceSide<P: Platform>: Send + Sync {
    /// One message from the host.
    fn on_message(&self, m: ToService<P>);

    /// One message from the host, sent with [`ServiceLink::send_here`] and delivered on the
    /// sender's thread, as the task's code: its work may run there, in its turn among the
    /// service's. By default, [`ServiceSide::on_message`].
    fn on_message_here(&self, m: ToService<P>) {
        self.on_message(m)
    }

    /// The task has ended (it died, or the host invalidated its connection): the service
    /// releases its state now, whoever still holds the service. By default, nothing.
    fn end(&self) {}
}
