//! One isolated task's life, [`TaskLife`]: running, then ended by a death in the task's own code
//! or by the host's invalidation; and the trap that makes such a death end the task instead of
//! the process.
//!
//! Every thread that runs the task's code does so inside [`TaskLife::run`], which sets the task's
//! trap on the thread: the task's connection (its messages and the handshake), each job on the
//! service queue, and each Metal completion handler the task adds ([`TaskLife::carry`]). A death
//! there (`raise::death::fatal`, an uncaught raise, a NULL guest access) writes its line, marks
//! the task dead and unwinds to that `run`, which returns `None`. What the guest then sees is the
//! host's: status 4 for the command and every later one.
//!
//! The task's other threads stop as a crashed process's did: at once, doing nothing more that
//! anyone sees. Each stops at its next step with an effect ([`stop_if_task_ended`]): a message
//! to the host, a Metal call (a commit, an encoded signal of a FIFO's event, a copy into a
//! resource), a write to guest memory, a job or a completion handler that has not begun. A
//! thread waiting on the host for a reply is woken and stops ([`TaskLife::waiting`]). The host's
//! invalidation ends the task the same way.
//!
//! A crash inside Metal is not the task's: a Metal exception no handler takes, a platform call
//! that ends the process, and a real crash in Metal, IOGPU or the driver end the process, as they
//! do outside isolation.
//!
//! A task that dies releases no Metal object: while a thread unwinds the task's code, and while
//! the end of a task that died releases its state, a platform handle dropped is leaked
//! ([`leaks_objects`]). The Rust state goes; the Metal objects stay as the death left them (an
//! encoder never ended, a command buffer half encoded), never released in a state Metal asserts
//! on. They hold their GPU memory for the life of the host's process: nothing takes them back.

#![expect(
    clippy::disallowed_methods,
    reason = "the task trap: a death in an isolated task's code unwinds to its run (6.4)"
)]

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::raise::RaiseId;
use crate::raise::death::DeathId;

/// Where a task is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Its threads run its jobs.
    Running,
    /// A death in its own code ended it.
    Died(DeathId),
    /// The host invalidated its connection.
    Exited,
}

/// The unwind payload that stops a thread of a task that has ended.
struct Stopped;

/// How a thread runs the task's code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// The task's work ([`TaskLife::run`]): it stops once the task has ended.
    Working,
    /// The task's end releasing its state ([`TaskLife::releasing`]): it goes on to the end.
    Releasing,
}

/// What runs once when the task dies: the transport's end of the connection.
type OnDeath = Box<dyn FnOnce() + Send>;

thread_local! {
    /// The task whose code this thread runs, and how, while a [`TaskLife::run`] or a
    /// [`TaskLife::releasing`] is open on it.
    static TRAP: RefCell<Option<(Arc<TaskLife>, Mode)>> = const { RefCell::new(None) };

    /// Platform handles dropped on this thread now are leaked ([`leaks_objects`]).
    static LEAKS: Cell<bool> = const { Cell::new(false) };

    /// `TRAP` holds a task. The checks every thread makes read this first, so a thread that never
    /// runs a task's code never touches `TRAP`, whose value has a destructor to register: a host
    /// thread, a Metal thread, or one already destroying its thread-locals.
    static IN_TASK: Cell<bool> = const { Cell::new(false) };
}

/// Wakes one thread that waits on the host for the task ([`TaskLife::waiting`]).
type Waker = Arc<dyn Fn() + Send + Sync>;

/// One isolated task's life. Shared by the transport and the task's service.
pub struct TaskLife {
    state: Mutex<State>,
    /// Set with the state's end, for the checks on every path out of the task.
    ended: AtomicBool,
    on_death: Mutex<Option<OnDeath>>,
    /// The threads waiting on the host now, each by its registration's number.
    wakers: Mutex<Vec<(u64, Waker)>>,
    next_waker: AtomicU64,
}

impl std::fmt::Debug for TaskLife {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskLife")
            .field("state", &*self.state.lock())
            .finish_non_exhaustive()
    }
}

impl TaskLife {
    /// A running task.
    pub fn new() -> Arc<TaskLife> {
        Arc::new(TaskLife {
            state: Mutex::new(State::Running),
            ended: AtomicBool::new(false),
            on_death: Mutex::new(None),
            wakers: Mutex::new(Vec::new()),
            next_waker: AtomicU64::new(0),
        })
    }

    /// The task whose code this thread runs, while a [`TaskLife::run`] or a
    /// [`TaskLife::releasing`] is open on it.
    pub fn current() -> Option<Arc<TaskLife>> {
        if !IN_TASK.get() {
            return None;
        }
        TRAP.with(|t| t.borrow().as_ref().map(|(l, _)| Arc::clone(l)))
    }

    /// Sets what runs once when the task dies, on the thread whose `run` caught the death,
    /// after its unwind. Nothing runs when the host ends the task.
    pub fn on_death(&self, f: impl FnOnce() + Send + 'static) {
        *self.on_death.lock() = Some(Box::new(f));
    }

    /// Whether the task has ended: died, or invalidated by the host.
    pub fn has_ended(&self) -> bool {
        self.ended.load(Ordering::SeqCst)
    }

    /// The death that ended the task, if one did.
    pub fn death(&self) -> Option<DeathId> {
        match *self.state.lock() {
            State::Died(id) => Some(id),
            State::Running | State::Exited => None,
        }
    }

    /// The host invalidated the task's connection: the task ends without a death. `false` when
    /// it had already ended.
    pub fn exit(&self) -> bool {
        let mut state = self.state.lock();
        if *state != State::Running {
            return false;
        }
        *state = State::Exited;
        self.ended.store(true, Ordering::SeqCst);
        drop(state);
        self.wake_waiters();
        true
    }

    /// Runs `f`, which waits on the host for the task, with `wake` registered: when the task
    /// ends, `wake` runs once, on the ending thread, and `f` is to see [`TaskLife::has_ended`]
    /// and give up. A host that never answers a task that goes on is waited for forever.
    pub fn waiting<R>(&self, wake: impl Fn() + Send + Sync + 'static, f: impl FnOnce() -> R) -> R {
        /// Takes the registration back when `f` returns or unwinds.
        struct Registered<'a>(&'a TaskLife, u64);
        impl Drop for Registered<'_> {
            fn drop(&mut self) {
                self.0.wakers.lock().retain(|(n, _)| *n != self.1);
            }
        }
        let n = self.next_waker.fetch_add(1, Ordering::Relaxed);
        self.wakers.lock().push((n, Arc::new(wake)));
        let _registered = Registered(self, n);
        f()
    }

    /// Wakes every thread waiting on the host, outside the lists' lock.
    fn wake_waiters(&self) {
        let waiters = std::mem::take(&mut *self.wakers.lock());
        for (_, wake) in waiters {
            wake();
        }
    }

    /// Runs `f` as the task's code: `None`, with `f` dropped unrun, once the task has ended, and
    /// `None` when a death in `f` ends it. A `run` inside another of the same task's on this
    /// thread is part of the outer one: the death unwinds to the outer.
    pub fn run<R>(self: &Arc<Self>, f: impl FnOnce() -> R) -> Option<R> {
        if self.has_ended() {
            // What `f` holds is the task's, and a task that died releases nothing.
            let leaks = LEAKS.get();
            LEAKS.set(self.death().is_some());
            drop(f);
            LEAKS.set(leaks);
            return None;
        }
        self.enter(Mode::Working, f)
    }

    /// Runs `f`, the task's end releasing its state, as the task's code after the task has
    /// ended: a message to the host there goes nowhere and the release goes on; a death there
    /// stops the release where it is (`None`), and the process goes on. When the task died, the
    /// platform handles the release drops are leaked.
    pub fn releasing<R>(self: &Arc<Self>, f: impl FnOnce() -> R) -> Option<R> {
        self.enter(Mode::Releasing, f)
    }

    /// Runs `f` with the task's trap set on this thread in `mode`.
    fn enter<R>(self: &Arc<Self>, mode: Mode, f: impl FnOnce() -> R) -> Option<R> {
        let nested = IN_TASK.get()
            && TRAP.with(|t| {
                t.borrow()
                    .as_ref()
                    .is_some_and(|(l, _)| Arc::ptr_eq(l, self))
            });
        if nested {
            return Some(f());
        }
        let outer = TRAP.with(|t| t.replace(Some((Arc::clone(self), mode))));
        let in_task = IN_TASK.replace(true);
        let leaks = LEAKS.get();
        LEAKS.set(mode == Mode::Releasing && self.death().is_some());
        let r = catch_unwind(AssertUnwindSafe(f));
        LEAKS.set(leaks);
        TRAP.with(|t| *t.borrow_mut() = outer);
        IN_TASK.set(in_task);
        match r {
            Ok(v) => Some(v),
            Err(p) if p.is::<Stopped>() => {
                self.report_death();
                None
            }
            // Not the task's: a bug, or a test's own capture.
            Err(p) => resume_unwind(p),
        }
    }

    /// `f` as a callback that runs as the task's code ([`TaskLife::run`]) when it is called: a
    /// Metal completion handler the task adds.
    pub fn carry<A, F>(self: &Arc<Self>, f: F) -> impl FnOnce(A) + Send + use<A, F>
    where
        A: 'static,
        F: FnOnce(A) + Send + 'static,
    {
        let life = Arc::clone(self);
        move |a: A| {
            let _ = life.run(move || f(a));
        }
    }

    /// Stops this thread here when it runs the task's work and the task has ended: its message
    /// to the host goes nowhere, and nothing after it runs. A thread already unwinding, one
    /// outside the task's `run` and the task's release go on.
    pub fn stop_if_ended(self: &Arc<Self>) {
        if !self.has_ended() || std::thread::panicking() || !IN_TASK.get() {
            return;
        }
        let working = TRAP.with(|t| {
            t.borrow()
                .as_ref()
                .is_some_and(|(l, mode)| Arc::ptr_eq(l, self) && *mode == Mode::Working)
        });
        if working {
            LEAKS.set(true);
            resume_unwind(Box::new(Stopped))
        }
    }

    /// The death `id` in the task's code, on a thread its trap covers: the first one ends the
    /// task and writes its line; every one unwinds to the thread's `run`.
    fn die(&self, id: DeathId, detail: Option<&str>) -> ! {
        let first = {
            let mut state = self.state.lock();
            let first = *state == State::Running;
            if first {
                *state = State::Died(id);
                self.ended.store(true, Ordering::SeqCst);
            }
            first
        };
        if first {
            self.wake_waiters();
            let mut line = match detail {
                Some(detail) => format!("reims-vgpu: isolated task died: {id}: {detail}"),
                None => format!("reims-vgpu: isolated task died: {id}"),
            };
            line.push('\n');
            use std::io::Write as _;
            // One write, so the line never interleaves with another thread's.
            let _ = std::io::stderr().write_all(line.as_bytes());
        }
        LEAKS.set(true);
        resume_unwind(Box::new(Stopped))
    }

    /// Runs the death hook, once, when the task died.
    fn report_death(&self) {
        if self.death().is_none() {
            return;
        }
        let hook = self.on_death.lock().take();
        if let Some(hook) = hook {
            hook();
        }
    }
}

/// Runs `f` on this thread outside the task's code whose trap covers it: host code a task's
/// thread runs in the link queue's place (a reply the link runs on the replying thread). A death
/// in `f` is the process's, as on a thread no task's trap covers, and `f` never stops for the
/// task's end. The trap comes back when `f` returns or unwinds.
pub(crate) fn outside_the_task<R>(f: impl FnOnce() -> R) -> R {
    /// Puts the thread's trap back.
    struct Restore {
        trap: Option<(Arc<TaskLife>, Mode)>,
        in_task: bool,
        leaks: bool,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            let trap = self.trap.take();
            TRAP.with(|t| *t.borrow_mut() = trap);
            IN_TASK.set(self.in_task);
            LEAKS.set(self.leaks);
        }
    }
    if !IN_TASK.get() {
        return f();
    }
    let _restore = Restore {
        trap: TRAP.with(|t| t.replace(None)),
        in_task: IN_TASK.replace(false),
        leaks: LEAKS.replace(false),
    };
    f()
}

/// A death that belongs to Metal, not to the task's own code: a Metal exception no handler takes,
/// or a platform call that ends the process.
fn inside_the_platform(id: &DeathId) -> bool {
    matches!(
        id,
        DeathId::Platform { .. }
            | DeathId::Raised {
                id: RaiseId::Platform,
                ..
            }
    )
}

/// `raise::death`'s hook: a death on a thread that runs an isolated task's code ends that task
/// and unwinds to the thread's [`TaskLife::run`]. Returns only when no task's trap covers the
/// thread, or the death is Metal's: the process then ends as it would outside isolation.
pub(crate) fn trap(id: DeathId, detail: Option<&str>) {
    if inside_the_platform(&id) {
        return;
    }
    if let Some(life) = TaskLife::current() {
        life.die(id, detail)
    }
}

/// Whether a platform handle dropped on this thread now is leaked instead of released: the thread
/// unwinds a task's code that died or was stopped, or releases the state of a task that died.
/// What a crashed process never released stays unreleased, and it has to: a death leaves Metal
/// objects in states whose release Metal asserts on (an encoder released without
/// `endEncoding` ends the process), and any of them may be anywhere in the task's state. So
/// a task that died keeps its Metal objects until the process exits.
pub fn leaks_objects() -> bool {
    LEAKS.get()
}

/// Stops this thread here when it runs the work of an isolated task that has ended
/// ([`TaskLife::stop_if_ended`]); anywhere else, nothing. Called before every step with an
/// effect: each Metal call, each write to guest memory.
pub fn stop_if_task_ended() {
    if !IN_TASK.get() {
        return;
    }
    let ended = TRAP.with(|t| match &*t.borrow() {
        Some((life, Mode::Working)) if life.has_ended() => Some(Arc::clone(life)),
        _ => None,
    });
    if let Some(life) = ended {
        life.stop_if_ended();
    }
}

/// Whether `payload` is a stopped task thread's unwind (for a harness that catches unwinds of its
/// own around a task's `run`).
pub fn is_stopped(payload: &(dyn Any + Send)) -> bool {
    payload.is::<Stopped>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::Raise;
    use crate::raise::death::{DeathKind, DeathSite};
    use crate::testkit::capture::catch_death;

    const OURS: DeathSite = DeathSite {
        kind: DeathKind::Assert,
        file: "life.test",
        line: 0,
    };

    /// A death inside `run` ends the task, not the thread: `run` answers `None`, the task
    /// records the death, and later runs do not start
    #[test]
    fn a_death_in_run_ends_the_task() {
        let life = TaskLife::new();
        assert_eq!(life.run(|| 5), Some(5));
        let r: Option<()> = life.run(|| crate::fatal!(OURS));
        assert_eq!(r, None);
        assert!(life.has_ended());
        assert_eq!(life.death().map(DeathId::site), Some(OURS));
        let ran = AtomicBool::new(false);
        assert_eq!(life.run(|| ran.store(true, Ordering::SeqCst)), None);
        assert!(!ran.load(Ordering::SeqCst));
    }

    /// The death hook runs once, after the dying thread's unwind; an exit runs nothing
    #[test]
    fn the_hook_runs_once_after_the_unwind() {
        let life = TaskLife::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        life.on_death(move || s2.lock().push("hook"));
        struct Local(Arc<Mutex<Vec<&'static str>>>);
        impl Drop for Local {
            fn drop(&mut self) {
                self.0.lock().push("local dropped");
            }
        }
        let s3 = seen.clone();
        let _ = life.run(move || {
            let _local = Local(s3);
            crate::fatal!(OURS)
        });
        let _ = life.run(|| crate::fatal!(OURS));
        assert_eq!(*seen.lock(), vec!["local dropped", "hook"]);
        assert!(!leaks_objects());
        let exited = TaskLife::new();
        let s4 = seen.clone();
        exited.on_death(move || s4.lock().push("never"));
        assert!(exited.exit());
        assert!(!exited.exit());
        assert!(exited.has_ended());
        assert_eq!(exited.death(), None);
        assert_eq!(seen.lock().len(), 2);
    }

    /// A nested run is part of the outer: the death unwinds to the outer run
    #[test]
    fn a_nested_run_unwinds_to_the_outer() {
        let life = TaskLife::new();
        let after_inner = AtomicBool::new(false);
        let r = life.run(|| {
            let _ = life.run(|| crate::fatal!(OURS));
            after_inner.store(true, Ordering::SeqCst);
        });
        assert_eq!(r, None);
        assert!(!after_inner.load(Ordering::SeqCst));
    }

    /// Outside a run a death is the process's; a Metal death is the process's inside one too
    #[test]
    fn deaths_outside_a_run_or_inside_metal_end_the_process() {
        assert_eq!(catch_death(|| crate::fatal!(OURS)), Err(OURS));
        let life = TaskLife::new();
        let metal = DeathId::Platform {
            kind: DeathKind::UncaughtObjc,
            call: "newBufferWithLength:options:",
        };
        let d = catch_death(|| life.run(|| crate::fatal!(metal)));
        assert_eq!(d.map(|_| ()), Err(metal.site()));
        let raised = catch_death(|| {
            life.run(|| {
                Raise::foreign("commit").fatal();
            })
        });
        assert!(raised.is_err());
        assert!(!life.has_ended());
    }

    /// Inside a run, code outside the task's dies as the process, leaves the task running, and
    /// does not stop for the task's end; the trap comes back after it
    #[test]
    fn a_death_outside_the_task_inside_its_run_ends_the_process() {
        let life = TaskLife::new();
        let d = catch_death(|| life.run(|| outside_the_task(|| crate::fatal!(OURS))));
        assert_eq!(d.map(|_| ()), Err(OURS));
        assert!(!life.has_ended());
        let l2 = life.clone();
        let ran = life.run(|| {
            assert!(TaskLife::current().is_some());
            let outside = outside_the_task(|| {
                assert!(TaskLife::current().is_none());
                l2.exit();
                // The task has ended: code outside it goes on.
                l2.stop_if_ended();
                7
            });
            assert!(Arc::ptr_eq(&TaskLife::current().unwrap(), &l2));
            outside
        });
        assert_eq!(ran, Some(7));
        assert!(TaskLife::current().is_none());
    }

    /// A carried callback runs as the task's code on any thread, and not once the task ended
    #[test]
    fn a_carried_callback_runs_as_the_task() {
        let life = TaskLife::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let cb = life.carry(move |v: u32| s2.lock().push(v));
        cb(1);
        let dies = life.carry(|_: u32| crate::fatal!(OURS));
        dies(2);
        assert!(life.has_ended());
        let s3 = seen.clone();
        let late = life.carry(move |v: u32| s3.lock().push(v));
        late(3);
        assert_eq!(*seen.lock(), vec![1]);
    }

    /// Once the task ended, a thread inside its run stops at `stop_if_ended`; one outside goes on
    #[test]
    fn stop_if_ended_stops_only_the_tasks_threads() {
        let life = TaskLife::new();
        life.stop_if_ended();
        let went_on = AtomicBool::new(false);
        assert!(life.exit());
        life.stop_if_ended();
        let other = TaskLife::new();
        let r = other.run(|| {
            life.stop_if_ended();
            went_on.store(true, Ordering::SeqCst);
        });
        assert_eq!(r, Some(()));
        assert!(went_on.load(Ordering::SeqCst));
        // Inside the task's own run, after its end began elsewhere: stops there.
        let running = TaskLife::new();
        let stopped_at = Arc::new(Mutex::new(0));
        let s2 = stopped_at.clone();
        let r = running.run(|| {
            *s2.lock() = 1;
            running.exit();
            running.stop_if_ended();
            *s2.lock() = 2;
        });
        assert_eq!(r, None);
        assert_eq!(*stopped_at.lock(), 1);
        assert!(is_stopped(&Stopped));
    }

    /// Handles dropped by a dying or stopped thread's unwind leak, and those the release of a
    /// dead task drops; the work, the unwind's catch and the release of an invalidated task
    /// release theirs
    #[test]
    fn the_unwind_and_a_dead_tasks_release_leak() {
        struct Probe(Arc<Mutex<Vec<bool>>>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.lock().push(leaks_objects());
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let life = TaskLife::new();
        let _ = life.run(|| drop(Probe(seen.clone())));
        let _ = life.run(|| {
            let _p = Probe(seen.clone());
            crate::fatal!(OURS)
        });
        assert!(!leaks_objects());
        let _ = life.releasing(|| drop(Probe(seen.clone())));
        let stopped = TaskLife::new();
        let _ = stopped.run(|| {
            let _p = Probe(seen.clone());
            stopped.exit();
            stopped.stop_if_ended();
        });
        let _ = stopped.releasing(|| drop(Probe(seen.clone())));
        assert_eq!(*seen.lock(), vec![false, true, true, true, false]);
        assert!(!leaks_objects());
    }

    /// Once the task ended, a thread of its work stops at its next step with an effect (a
    /// platform call, a guest write); a thread of another task, one outside any, and the release
    /// go on
    #[test]
    fn a_step_with_an_effect_stops_an_ended_tasks_work() {
        use crate::guest::GuestPtr;
        use crate::testkit::guest_ram::RawRegion;

        let region = RawRegion::new(64).unwrap();
        let word = GuestPtr::new(region.addr());
        let steps = Arc::new(Mutex::new(Vec::new()));
        let life = TaskLife::new();
        let s2 = steps.clone();
        let r = life.run(|| {
            word.write_u32(0, 1, OURS);
            s2.lock().push("written");
            assert!(life.exit());
            word.write_u32(0, 2, OURS);
            s2.lock().push("never");
        });
        assert_eq!(r, None);
        assert_eq!(*steps.lock(), vec!["written"]);
        assert_eq!(region.read_u32(0), Some(1));
        // Outside the task's work: the write lands.
        stop_if_task_ended();
        word.write_u32(0, 3, OURS);
        let other = TaskLife::new();
        assert_eq!(other.run(|| word.write_u32(0, 4, OURS)), Some(()));
        assert_eq!(life.releasing(|| word.write_u32(0, 5, OURS)), Some(()));
        assert_eq!(region.read_u32(0), Some(5));
        // A platform call: the fake's script is the metal crate's message layer.
        let p = crate::testkit::FakePlatform::new();
        let dead = TaskLife::new();
        let r = dead.run(|| {
            dead.exit();
            let _ = p.world().script.next(crate::platform::ObjId(0), "commit");
        });
        assert_eq!(r, None);
    }

    /// A thread waiting on the host is woken once when the task ends, by a death or by the
    /// host; a registration that ended is never woken
    #[test]
    fn waiters_are_woken_by_the_end() {
        let woken = Arc::new(Mutex::new(Vec::new()));
        let life = TaskLife::new();
        let w = woken.clone();
        life.waiting(move || w.lock().push("gone"), || ());
        let w = woken.clone();
        let (l2, w2) = (life.clone(), woken.clone());
        life.waiting(
            move || w.lock().push("exit"),
            || {
                assert!(l2.exit());
                w2.lock().push("after");
            },
        );
        assert_eq!(*woken.lock(), vec!["exit", "after"]);
        let dying = TaskLife::new();
        let w = woken.clone();
        let d2 = dying.clone();
        let r: Option<()> =
            dying.run(|| d2.waiting(move || w.lock().push("death"), || crate::fatal!(OURS)));
        assert_eq!(r, None);
        assert_eq!(*woken.lock(), vec!["exit", "after", "death"]);
    }

    /// A run that does not start because the task died drops its work as the task's: leaked
    #[test]
    fn a_dead_tasks_skipped_run_leaks() {
        struct Probe(Arc<Mutex<Vec<bool>>>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.lock().push(leaks_objects());
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let died = TaskLife::new();
        let _ = died.run(|| crate::fatal!(OURS));
        let p = Probe(seen.clone());
        assert_eq!(died.run(move || drop(p)), None);
        let exited = TaskLife::new();
        exited.exit();
        let p = Probe(seen.clone());
        assert_eq!(exited.run(move || drop(p)), None);
        assert_eq!(*seen.lock(), vec![true, false]);
        assert!(!leaks_objects());
    }

    /// The release runs after the end, goes on past a message to the host, and a death in it
    /// stops only the release
    #[test]
    fn the_release_goes_on_to_its_end_and_a_death_stops_only_it() {
        let life = TaskLife::new();
        assert!(life.exit());
        let steps = Arc::new(Mutex::new(Vec::new()));
        let s2 = steps.clone();
        let r = life.releasing(|| {
            s2.lock().push(1);
            life.stop_if_ended();
            s2.lock().push(2);
            assert!(TaskLife::current().is_some_and(|l| Arc::ptr_eq(&l, &life)));
        });
        assert_eq!(r, Some(()));
        assert_eq!(*steps.lock(), vec![1, 2]);
        let r: Option<()> = life.releasing(|| crate::fatal!(OURS));
        assert_eq!(r, None);
        // The task ended by the host's invalidation, not by the release's death.
        assert_eq!(life.death(), None);
        assert!(TaskLife::current().is_none());
    }
}
