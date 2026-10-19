//! Isolated tasks in this process: [`TaskThreads`], the launcher that runs each isolated task's
//! service ([`GpuTask`]) with the service's semantics on threads of its own, and each task's
//! life ([`TaskLife`]).
//!
//! Under the device's process isolation every user task is a remote task, and its commands
//! cross a connection to its own service ([`link`]): a fresh service per task, never pooled or
//! reused, whose objects come from the host's Metal device and which reads guest RAM through the
//! host's own mapping of it. The service's threads are the task's connection queue (the
//! handshake and its messages, in order), the service's own queue (its jobs) and Metal's
//! completion threads; an exec's job runs on the FIFO thread that sent it, in its turn on both
//! queues ([`link`], `ServiceLink::send_here`).
//!
//! What the host sees of a task:
//! - **A death in the task's own code** ends the task and nothing else ([`life`]): its threads
//!   stop at once, each at its next step with an effect, so none writes guest memory, sends to
//!   Metal or to the host after the death; its state is released (its Metal objects leaked), and
//!   then the host's endpoint hears of the interruption, which completes every outstanding token
//!   with status 4 in ascending id order. Every later command on the task completes with 4 at
//!   once, and the first send after the death invalidates the connection. A crash inside Metal
//!   ends the process.
//! - **The invalidation** (the remote task's drop, after its drain) ends the task the same way
//!   without a death, releasing its state before the drop goes on.
//!
//! Ordering contracts (each named test pins it):
//! - messages reach the service in send order, and replies and the service's messages reach the
//!   host in the order the service sent them (`messages_and_replies_keep_their_order`);
//! - a death releases the service's state before the host hears of it, and later sends are
//!   dropped with their replies released at once (`a_death_ends_the_task_then_tells_the_host`);
//! - the invalidation releases the service's state before it returns
//!   (`invalidation_releases_the_service_first`);
//! - once the task has ended, its threads stop at their next step with an effect
//!   (`a_step_with_an_effect_stops_an_ended_tasks_work`), and one waiting on the host is woken
//!   (`a_retain_the_host_never_answers_stops_at_the_end`).

pub mod life;
mod link;

use std::sync::Arc;

use crate::gpu_task::GpuTask;
use crate::platform::Platform;
use crate::tasks::launcher::{HostLink, HostSide, ServiceLink, ServiceSide, TaskLauncher};

pub use life::TaskLife;

/// Makes the service of one launch over the host's platform, with its end of the connection and
/// its life: `GpuTask::new` in the product, a scripted service in a test.
pub type ServiceFactory<P> = Arc<
    dyn Fn(&Arc<P>, Arc<dyn HostLink<P>>, &Arc<TaskLife>) -> Arc<dyn ServiceSide<P>> + Send + Sync,
>;

/// The product's task launcher: one fresh service per isolated task, on threads of its own.
pub struct TaskThreads<P: Platform> {
    factory: Option<ServiceFactory<P>>,
}

impl<P: Platform> std::fmt::Debug for TaskThreads<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskThreads")
            .field("scripted", &self.factory.is_some())
            .finish()
    }
}

impl<P: Platform> Default for TaskThreads<P> {
    fn default() -> Self {
        TaskThreads::new()
    }
}

impl<P: Platform> TaskThreads<P> {
    /// A launcher whose services are `GpuTask`s.
    pub fn new() -> TaskThreads<P> {
        TaskThreads { factory: None }
    }

    /// A launcher whose services `factory` makes.
    pub fn with_factory(factory: ServiceFactory<P>) -> TaskThreads<P> {
        TaskThreads {
            factory: Some(factory),
        }
    }

    /// [`TaskLauncher::launch`], answering the task's life too.
    pub fn launch_task(
        &self,
        platform: &Arc<P>,
        host: Arc<dyn HostSide<P>>,
    ) -> (Arc<dyn ServiceLink<P>>, Arc<TaskLife>) {
        let life = TaskLife::new();
        let conn = link::Connection::new(platform, host, Arc::clone(&life));
        let to_host: Arc<dyn HostLink<P>> = Arc::new(link::ToHostLink(Arc::clone(&conn)));
        let service = match &self.factory {
            Some(factory) => factory(platform, to_host, &life),
            None => {
                let service: Arc<dyn ServiceSide<P>> =
                    GpuTask::new(Arc::clone(platform), to_host, Arc::clone(&life));
                service
            }
        };
        conn.serve(service);
        (Arc::new(link::Link(conn)), life)
    }
}

impl<P: Platform> TaskLauncher<P> for TaskThreads<P> {
    fn launch(&self, platform: &Arc<P>, host: Arc<dyn HostSide<P>>) -> Arc<dyn ServiceLink<P>> {
        self.launch_task(platform, host).0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // tests poll in real time

    use super::*;
    use crate::raise::death::{DeathId, DeathKind, DeathSite};
    use crate::tasks::wire::{Reply, ToHost, ToService};
    use crate::testkit::FakePlatform;
    use parking_lot::Mutex;
    use std::time::{Duration, Instant};

    type P = FakePlatform;

    /// Where the echo service dies: its own check.
    const ECHO_DEATH: DeathSite = DeathSite {
        kind: DeathKind::Abort,
        file: "echo",
        line: 0,
    };

    /// The host side: records every call.
    #[derive(Default)]
    struct Host {
        events: Mutex<Vec<String>>,
    }

    impl HostSide<P> for Host {
        fn on_message(&self, m: ToHost<P>) {
            self.events.lock().push(m.selector().to_owned());
        }
        fn on_interrupted(&self) {
            self.events.lock().push("interrupted".into());
        }
        fn on_invalidated(&self) {
            self.events.lock().push("invalidated".into());
        }
    }

    /// A service that echoes: FlushResources replies 1 after a message to the host; UnmapMemory
    /// dies; ReplacePhysical replies twice; GetComputeInfo never replies; `end` is recorded.
    struct Echo {
        host: Arc<dyn HostLink<P>>,
        ended: Arc<Mutex<Vec<String>>>,
    }

    impl ServiceSide<P> for Echo {
        fn on_message(&self, m: ToService<P>) {
            match m {
                ToService::FlushResources { reply } => {
                    self.host.send(ToHost::ReleaseIoSurface { mapping_id: 7 });
                    reply.call(1);
                }
                ToService::UnmapMemory { .. } => crate::fatal!(ECHO_DEATH),
                ToService::ReplacePhysical { physical, reply } => {
                    reply.call(physical);
                    reply.call(physical.wrapping_add(1));
                }
                ToService::GetComputeInfo { reply, .. } => std::mem::forget(reply),
                ToService::DiscardResources { reply, .. } => {
                    reply.call(1);
                    self.host.send(ToHost::ReleaseIoSurface { mapping_id: 8 });
                }
                _ => {}
            }
        }
        fn end(&self) {
            self.ended.lock().push("service ended".into());
        }
    }

    struct Rig {
        host: Arc<Host>,
        link: Arc<dyn ServiceLink<P>>,
        life: Arc<TaskLife>,
        ended: Arc<Mutex<Vec<String>>>,
    }

    fn rig() -> Rig {
        let p = Arc::new(FakePlatform::new());
        let ended = Arc::new(Mutex::new(Vec::new()));
        let e2 = ended.clone();
        let l = TaskThreads::<P>::with_factory(Arc::new(move |_p, host, _life| {
            Arc::new(Echo {
                host,
                ended: e2.clone(),
            })
        }));
        let host = Arc::new(Host::default());
        let (link, life) = l.launch_task(&p, host.clone());
        Rig {
            host,
            link,
            life,
            ended,
        }
    }

    fn wait_until(f: impl Fn() -> bool) {
        let start = Instant::now();
        while !f() {
            assert!(start.elapsed() < Duration::from_secs(10), "timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A reply that records its runs and holds `sentinel` (standing for the remote task).
    fn counted(sentinel: &Arc<()>, runs: &Arc<Mutex<Vec<u32>>>) -> Reply<u32> {
        let (s, r) = (sentinel.clone(), runs.clone());
        Reply::new(move |v| {
            let _ = &s;
            r.lock().push(v);
        })
    }

    /// Messages cross in order; a reply runs once, after the messages the service sent before
    /// it, and is released right after
    #[test]
    fn messages_and_replies_keep_their_order() {
        let r = rig();
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sentinel = Arc::new(());
        let (h2, r2) = (r.host.clone(), runs.clone());
        r.link.send(ToService::FlushResources {
            reply: Reply::new(move |v| {
                // The service's message came first.
                assert_eq!(h2.events.lock().len(), 1);
                r2.lock().push(v);
            }),
        });
        r.link.send(ToService::ReplacePhysical {
            physical: 5,
            reply: counted(&sentinel, &runs),
        });
        let weak = Arc::downgrade(&sentinel);
        drop(sentinel);
        wait_until(|| weak.upgrade().is_none());
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(*runs.lock(), vec![1, 5]);
        assert_eq!(
            *r.host.events.lock(),
            vec!["releaseIOSurfaceWithMappingID:".to_owned()]
        );
        assert!(!r.life.has_ended());
    }

    /// A death in the service ends the task: the service's state goes, then the host hears of
    /// the interruption, then pending replies are released unrun; a later send is dropped with
    /// its reply released at once, and invalidates the connection
    #[test]
    fn a_death_ends_the_task_then_tells_the_host() {
        let r = rig();
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sentinel = Arc::new(());
        r.link.send(ToService::GetComputeInfo {
            pipeline_state_reference: 0,
            max_key: 0,
            count: 0,
            offset: 0,
            reply: counted(&sentinel, &runs),
        });
        let weak = Arc::downgrade(&sentinel);
        drop(sentinel);
        let (e2, h2) = (r.ended.clone(), r.host.clone());
        r.link.send(ToService::UnmapMemory {
            offset: 0,
            length: 0,
            reply: Reply::new(move |_| {}),
        });
        wait_until(|| weak.upgrade().is_none());
        assert_eq!(r.life.death().map(DeathId::site), Some(ECHO_DEATH));
        assert_eq!(*e2.lock(), vec!["service ended".to_owned()]);
        assert_eq!(*h2.events.lock(), vec!["interrupted".to_owned()]);
        assert!(runs.lock().is_empty());
        // A later send: dropped, its reply released at once; the connection is invalidated.
        let s2 = Arc::new(());
        let w2 = Arc::downgrade(&s2);
        r.link.send(ToService::FlushResources {
            reply: counted(&s2, &runs),
        });
        drop(s2);
        assert!(w2.upgrade().is_none());
        wait_until(|| r.host.events.lock().len() == 2);
        assert_eq!(r.host.events.lock()[1], "invalidated");
        assert!(runs.lock().is_empty());
    }

    /// The invalidation ends the task without a death: the service's state goes before it
    /// returns, the host hears of it once, and a pending reply is released unrun after that
    #[test]
    fn invalidation_releases_the_service_first() {
        let r = rig();
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sentinel = Arc::new(());
        r.link.send(ToService::GetComputeInfo {
            pipeline_state_reference: 0,
            max_key: 0,
            count: 0,
            offset: 0,
            reply: counted(&sentinel, &runs),
        });
        let weak = Arc::downgrade(&sentinel);
        drop(sentinel);
        r.link.invalidate();
        assert_eq!(*r.ended.lock(), vec!["service ended".to_owned()]);
        r.link.invalidate();
        wait_until(|| weak.upgrade().is_none());
        assert_eq!(*r.host.events.lock(), vec!["invalidated".to_owned()]);
        assert!(r.life.has_ended());
        assert_eq!(r.life.death(), None);
        assert!(runs.lock().is_empty());
    }

    /// Once the task has ended, the service's messages to the host go nowhere
    #[test]
    fn a_dead_task_sends_nothing() {
        let r = rig();
        r.link.send(ToService::UnmapMemory {
            offset: 0,
            length: 0,
            reply: Reply::new(|_| {}),
        });
        wait_until(|| r.host.events.lock().len() == 1);
        // The service's own discard would reply and send; the task is dead, so neither.
        r.link.send(ToService::DiscardResources {
            resources: Vec::new(),
            count: 0,
            reply: Reply::new(|_| {}),
        });
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(
            *r.host.events.lock(),
            vec!["interrupted".to_owned(), "invalidated".to_owned()]
        );
    }

    /// A surface the host lends reaches the service as another object over the same memory:
    /// what the service holds never counts on the host's object, and the host's reference in its
    /// answer goes when the answer returns
    #[test]
    fn a_lent_surface_is_the_services_own_object() {
        use crate::platform::Handle;
        use crate::platform::os::surface::Surface;
        use crate::testkit::surfaces::FakeSurface;

        struct Lender(FakeSurface);
        impl HostSide<P> for Lender {
            fn on_message(&self, m: ToHost<P>) {
                if let ToHost::RetainIoSurface { reply, .. } = m {
                    reply.call(Some(self.0.clone_ref()));
                }
            }
            fn on_interrupted(&self) {}
            fn on_invalidated(&self) {}
        }
        struct Borrower {
            host: Arc<dyn HostLink<P>>,
            got: Arc<Mutex<Option<FakeSurface>>>,
        }
        impl ServiceSide<P> for Borrower {
            fn on_message(&self, m: ToService<P>) {
                if let ToService::FlushResources { reply } = m {
                    let got = self.got.clone();
                    self.host.send(ToHost::RetainIoSurface {
                        mapping_id: 1,
                        reply: Reply::new(move |s| *got.lock() = s),
                    });
                    reply.call(1);
                }
            }
        }

        let p = Arc::new(FakePlatform::new());
        let lent = p.fake_surfaces().make_simple(4, 4, 0, 4);
        let got = Arc::new(Mutex::new(None));
        let g2 = got.clone();
        let l = TaskThreads::<P>::with_factory(Arc::new(move |_p, host, _life| {
            Arc::new(Borrower {
                host,
                got: g2.clone(),
            })
        }));
        let (link, _life) = l.launch_task(&p, Arc::new(Lender(lent.clone_ref())));
        link.send(ToService::FlushResources {
            reply: Reply::new(|_| {}),
        });
        wait_until(|| got.lock().is_some());
        let own = got.lock().take().unwrap();
        assert_ne!(own.id(), lent.id());
        assert_eq!((own.width(), own.height()), (4, 4));
        assert!(Arc::ptr_eq(&own.memory().unwrap(), &lent.memory().unwrap()));
        let held = own.clone_ref();
        // The lender's own handle and the test's: the service's two count on its own object.
        assert_eq!(
            (Surface::retain_count(&lent), Surface::retain_count(&own)),
            (2, 2)
        );
        drop((own, held));
        link.invalidate();
    }

    /// Each launch is a fresh task with a life of its own
    #[test]
    fn each_launch_is_fresh() {
        let a = rig();
        let b = rig();
        assert!(!Arc::ptr_eq(&a.life, &b.life));
        assert!(format!("{:?}", TaskThreads::<P>::new()).contains("scripted: false"));
    }
}
