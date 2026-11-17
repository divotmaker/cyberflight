//! FRP shot intake: a thread that dials the bus, announces cyberflight as a
//! collector, aggregates shots and stamps each with its arrival time.

use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;

use cf_scene::shot::ReceivedShot;
use flightrelay::types::roles;
use flightrelay::{
    BusActor, BusState, DEFAULT_HEARTBEAT, EndpointConfig, FrpBody, FrpConnection, FrpEndpoint,
    FrpMessage, ShotAggregator, ShotEvent, Transport,
};

/// FRP spec version we support.
const FRP_VERSION: &str = flightrelay::SPEC_VERSION;

/// How often a lost or refused FRP connection is retried, and how long a
/// dial (connect and WebSocket upgrade) may take.
pub const RECONNECT_INTERVAL: Duration = Duration::from_secs(5);

/// How often the intake forgets actors that stopped heartbeating.
const EXPIRE_INTERVAL: Duration = Duration::from_millis(500);

/// The display name of cyberflight's actor on the bus.
const ACTOR_NAME: &str = "cyberflight";

/// The FRP device dialled without `--frp`: flighthook on this machine.
///
/// An IPv4 address rather than `localhost`: by default flighthook listens on
/// IPv4 only, and where `localhost` resolves to `::1` first (Windows), every
/// dial waits for the refused IPv6 attempt (about 2 s) before trying IPv4.
pub const DEFAULT_FRP_URL: &str = "ws://127.0.0.1:5880/frp";

/// The FRP device URL: the value after `--frp` in `args`, or
/// [`DEFAULT_FRP_URL`] without the flag.
///
/// # Errors
///
/// Returns an error if `--frp` has no value or the value is not a
/// `ws://` or `wss://` URL.
pub fn frp_url(args: &[String]) -> anyhow::Result<String> {
    let Some(i) = args.iter().position(|a| a == "--frp") else {
        return Ok(DEFAULT_FRP_URL.to_owned());
    };
    let url = args
        .get(i + 1)
        .ok_or_else(|| anyhow::anyhow!("--frp needs a URL, e.g. --frp ws://<host>:5880/frp"))?;
    if url.starts_with("ws://") || url.starts_with("wss://") {
        Ok(url.clone())
    } else {
        anyhow::bail!("--frp {url}: expected a ws:// or wss:// URL, e.g. ws://<host>:5880/frp")
    }
}

/// Open the FRP endpoint cyberflight dials at `url`.
///
/// Cyberflight is an FRP *collector*: an actor on the bus that receives
/// shots and sends nothing but its own `bus.actor`. A dial gives up after
/// [`RECONNECT_INTERVAL`], so a peer that accepts and never answers is a
/// failed dial. Opening a dialing endpoint performs no I/O, so this only
/// fails on a malformed configuration.
///
/// # Errors
///
/// Returns an error if the endpoint cannot be opened.
pub fn frp_endpoint(url: &str) -> Result<FrpEndpoint, flightrelay::FrpError> {
    FrpEndpoint::open(
        EndpointConfig::new(Transport::connect(url)).with_handshake_timeout(RECONNECT_INTERVAL),
    )
}

/// Cyberflight's `bus.actor`: role `collector`, heartbeat
/// [`DEFAULT_HEARTBEAT`].
fn actor_message(actor_id: &str) -> FrpMessage {
    FrpMessage::new(
        actor_id,
        BusActor::new(ACTOR_NAME, DEFAULT_HEARTBEAT.as_secs_f64()).with_role(roles::COLLECTOR),
    )
}

/// Turns FRP messages into shots: tracks the bus's launch monitors and
/// their ready states, and aggregates each shot's messages.
///
/// Every message is fed, whatever is on screen, so a shot that arrives while
/// an earlier ball is still flying or rolling is never lost.
#[derive(Debug, Default)]
pub struct ShotIntake {
    shots: ShotAggregator,
    bus: BusState,
    /// Ready state (no blockers) per launch monitor on the bus, by the name
    /// in its `bus.actor`.
    pub lm_states: HashMap<String, bool>,
}

impl ShotIntake {
    /// Process one FRP message, read at `now`. Returns a shot once its
    /// `shot.finished` arrives and it converts to a simulation input.
    ///
    /// A message for a shot that already finished is not shown again: one
    /// from the shot's own launch monitor is logged as a warning, and an
    /// addendum from another actor (e.g. a slow flight model's
    /// `shot.flight`) is noted in the log. The shot has already been shown.
    pub fn ingest_at(&mut self, msg: &FrpMessage, now: Instant) -> Option<ReceivedShot> {
        if self.bus.apply_at(msg, now) {
            self.refresh_lm_states();
        }
        let completed = match self.shots.feed_at(msg, now)? {
            ShotEvent::Finished(shot) => shot,
            ShotEvent::Updated(shot) => {
                eprintln!(
                    "INFO: shot #{} from {} updated after it finished; the shot on screen is unchanged",
                    shot.key.shot_number, shot.from
                );
                return None;
            }
            ShotEvent::Late(late) => {
                eprintln!("WARN: {late}");
                return None;
            }
        };
        match ReceivedShot::from_frp(&completed) {
            Ok(shot) => {
                for clamp in &shot.clamped {
                    eprintln!(
                        "WARN: shot #{} from {}: {clamp}",
                        shot.shot_number, shot.device
                    );
                }
                Some(shot)
            }
            Err(reason) => {
                eprintln!(
                    "WARN: shot #{} from {} not shown: {reason}",
                    completed.key.shot_number, completed.from
                );
                None
            }
        }
    }

    /// Forget the actors that had missed three heartbeats at `now`, with
    /// their state and open shots.
    pub fn expire_at(&mut self, now: Instant) {
        let gone = self.bus.expire_at(now);
        for actor in &gone {
            self.shots.forget_actor_at(actor, now);
        }
        if !gone.is_empty() {
            self.refresh_lm_states();
        }
    }

    /// Rebuild [`lm_states`](Self::lm_states) from the bus: one entry per
    /// `launch_monitor` actor, ready when it lists no blockers. Two actors
    /// with one name are told apart by their actor ids.
    fn refresh_lm_states(&mut self) {
        let monitors: Vec<(&str, &BusActor)> = self
            .bus
            .actors()
            .filter(|(_, actor)| actor.has_role(roles::LAUNCH_MONITOR))
            .collect();
        self.lm_states = monitors
            .iter()
            .map(|&(id, actor)| {
                let shared = monitors
                    .iter()
                    .filter(|(_, other)| other.name == actor.name)
                    .count()
                    > 1;
                let label = if shared {
                    format!("{} ({id})", actor.name)
                } else {
                    actor.name.clone()
                };
                (label, actor.is_clear())
            })
            .collect();
    }
}

/// A shot and when it arrived.
#[derive(Debug)]
pub struct ArrivedShot {
    pub shot: ReceivedShot,
    /// When its last message arrived, in seconds on the app's clock (since
    /// the `start` given to [`FrpState::new`]).
    pub arrival: f64,
}

/// What the intake thread tells the app.
#[derive(Debug)]
pub enum IntakeEvent {
    /// A dial has started: not connected.
    Connecting,
    /// The connection is up and cyberflight's actor announced.
    Connected,
    /// The connection was lost.
    Disconnected,
    /// The launch monitor ready states changed.
    LmStates(HashMap<String, bool>),
    Shot(ArrivedShot),
}

/// Shot intake from an FRP bus, used by both windowed and streaming modes.
///
/// A dedicated thread holds the connection and reads it whatever the app is
/// doing: it connects, announces cyberflight as a collector actor (a new
/// actor id each run) and republishes its `bus.actor` every
/// [`DEFAULT_HEARTBEAT`], reconnects every [`RECONNECT_INTERVAL`] while
/// disconnected, aggregates shots and stamps each with its arrival time.
/// [`poll`](Self::poll) never blocks.
///
/// The thread is never joined. It ends at its next report after this is
/// dropped (a shot, a ready-state change, a disconnect or a dial), or with
/// the process.
pub struct FrpState {
    events: mpsc::Receiver<IntakeEvent>,
    /// Whether the bus is connected.
    pub connected: bool,
    /// Last known ready state per launch monitor, by name; empty while
    /// disconnected.
    pub lm_states: HashMap<String, bool>,
    /// The intake thread has ended (reported once).
    stopped: bool,
}

impl FrpState {
    /// Start taking in shots from the bus at `url`. Arrival times are
    /// seconds since `start`; `wake` is called after each shot, connection
    /// change or ready-state change, from the intake thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the FRP endpoint cannot be opened or the thread
    /// cannot be started.
    pub fn new(
        url: &str,
        start: Instant,
        wake: impl Fn() + Send + 'static,
    ) -> anyhow::Result<Self> {
        let actor_id = flightrelay::new_actor_id();
        eprintln!("FRP actor id {actor_id} (new for this run)");
        let (events, _thread) = spawn_intake(url, &actor_id, start, wake, RECONNECT_INTERVAL)?;
        Ok(Self::from_events(events))
    }

    /// The app's end of an intake that reports on `events`.
    pub fn from_events(events: mpsc::Receiver<IntakeEvent>) -> Self {
        Self {
            events,
            connected: false,
            lm_states: HashMap::new(),
            stopped: false,
        }
    }

    /// Apply what the intake thread reported since the last call. Returns
    /// the shots that arrived, oldest first.
    pub fn poll(&mut self) -> Vec<ArrivedShot> {
        let mut shots = Vec::new();
        loop {
            match self.events.try_recv() {
                Ok(IntakeEvent::Connecting | IntakeEvent::Disconnected) => {
                    self.connected = false;
                    self.lm_states.clear();
                }
                Ok(IntakeEvent::Connected) => self.connected = true,
                Ok(IntakeEvent::LmStates(states)) => self.lm_states = states,
                Ok(IntakeEvent::Shot(shot)) => shots.push(shot),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !self.stopped {
                        eprintln!("WARN: the FRP intake has stopped; no more shots are taken in");
                        self.stopped = true;
                    }
                    self.connected = false;
                    self.lm_states.clear();
                    break;
                }
            }
        }
        shots
    }
}

/// Start the intake thread for the bus at `url`, as the actor `actor_id`,
/// dialing no more often than every `reconnect`. Returns its event channel
/// and handle.
fn spawn_intake(
    url: &str,
    actor_id: &str,
    start: Instant,
    wake: impl Fn() + Send + 'static,
    reconnect: Duration,
) -> anyhow::Result<(mpsc::Receiver<IntakeEvent>, thread::JoinHandle<()>)> {
    let endpoint = frp_endpoint(url).context("open the FRP endpoint")?;
    let (tx, rx) = mpsc::channel();
    let url = url.to_owned();
    let me = actor_message(actor_id);
    let handle = thread::Builder::new()
        .name("frp-intake".to_owned())
        .spawn(move || run_intake(&url, &me, endpoint, start, &tx, &wake, reconnect))
        .context("start the FRP intake thread")?;
    Ok((rx, handle))
}

/// The intake thread: dial, read until the connection is lost, repeat, as
/// [`Dialer`] decides. Returns once the app has dropped its end of `tx`.
fn run_intake(
    url: &str,
    me: &FrpMessage,
    mut endpoint: FrpEndpoint,
    start: Instant,
    tx: &mpsc::Sender<IntakeEvent>,
    wake: &dyn Fn(),
    reconnect: Duration,
) {
    // Report `event` and wake the app; false once the app is gone.
    let report = |event: IntakeEvent| {
        let sent = tx.send(event).is_ok();
        if sent {
            wake();
        }
        sent
    };
    let mut dialer = Dialer::new(reconnect);
    loop {
        match dialer.step(Instant::now()) {
            DialStep::Wait(wait) => thread::sleep(wait),
            DialStep::Dial => {
                // Nothing to wake the app for: it is not connected either way.
                if tx.send(IntakeEvent::Connecting).is_err() {
                    return;
                }
                eprintln!("Connecting to FRP at {url}...");
                match endpoint.establish() {
                    Ok(mut conn) => {
                        if !read_connection(&mut conn, me, start, &report) {
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!("FRP not available at {url} (expected {FRP_VERSION}): {e}");
                    }
                }
            }
        }
    }
}

/// Greet the bus on `conn` and read it until the connection is lost,
/// telling the app through `report`. Returns false once the app is gone.
fn read_connection(
    conn: &mut FrpConnection,
    me: &FrpMessage,
    start: Instant,
    report: &dyn Fn(IntakeEvent) -> bool,
) -> bool {
    if let Err(e) = conn.greet(std::slice::from_ref(me), &[]) {
        eprintln!("FRP disconnected: {e}");
        return true;
    }
    eprintln!("Connected (FRP {FRP_VERSION})");
    if !report(IntakeEvent::Connected) {
        return false;
    }

    let mut reader = IntakeReader::new(start, Instant::now());
    let reason = 'read: loop {
        let read = conn.recv_timeout(reader.wait(Instant::now()));
        let now = Instant::now();
        let msg = match read {
            // `None`: nothing arrived in time.
            Ok(msg) => msg,
            // Not an FRP message: ignore it and keep reading.
            Err(flightrelay::FrpError::Json(_)) => None,
            Err(e) => break e,
        };
        for action in reader.read(msg.as_ref(), now) {
            match action {
                ReadAction::Announce => {
                    if let Err(e) = conn.send(me) {
                        break 'read e;
                    }
                }
                ReadAction::Report(event) => {
                    if !report(event) {
                        return false;
                    }
                }
            }
        }
    };
    // Device states and partly received shots go with the connection.
    eprintln!("FRP disconnected: {reason}");
    report(IntakeEvent::Disconnected)
}

/// What the intake does next while it is not connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DialStep {
    /// Dial now.
    Dial,
    /// Sleep this long first.
    Wait(Duration),
}

/// When the intake dials, on the time it is given: at once the first time,
/// then no sooner than `reconnect` after the start of the previous dial.
#[derive(Debug)]
struct Dialer {
    reconnect: Duration,
    /// When the last dial started.
    last: Option<Instant>,
}

impl Dialer {
    fn new(reconnect: Duration) -> Self {
        Self {
            reconnect,
            last: None,
        }
    }

    /// What to do at `now`.
    fn step(&mut self, now: Instant) -> DialStep {
        if let Some(last) = self.last {
            let since = now.saturating_duration_since(last);
            if since < self.reconnect {
                return DialStep::Wait(self.reconnect - since);
            }
        }
        self.last = Some(now);
        DialStep::Dial
    }
}

/// What the intake does after a read, in the order given.
// A few of these live for one pass of the read loop; the event moves on to
// the app's channel unboxed, as it is sent.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum ReadAction {
    /// Send cyberflight's `bus.actor`.
    Announce,
    /// Tell the app.
    Report(IntakeEvent),
}

/// The read loop of one connection, on the time it is given: aggregates
/// shots, stamps each with its arrival, republishes cyberflight's
/// `bus.actor` every [`DEFAULT_HEARTBEAT`] and on `bus.sync`, forgets
/// silent actors every [`EXPIRE_INTERVAL`] and reports ready-state changes.
#[derive(Debug)]
struct IntakeReader {
    intake: ShotIntake,
    /// The ready states the app has; it clears them when a dial starts.
    reported: HashMap<String, bool>,
    /// The app's clock: arrival times are seconds since this.
    start: Instant,
    /// When `bus.actor` was last republished on the heartbeat.
    beat: Instant,
    /// When silent actors were last expired.
    expired: Instant,
}

impl IntakeReader {
    /// A reader for a connection that was greeted at `now`, stamping
    /// arrivals in seconds since `start`.
    fn new(start: Instant, now: Instant) -> Self {
        Self {
            intake: ShotIntake::default(),
            reported: HashMap::new(),
            start,
            beat: now,
            expired: now,
        }
    }

    /// How long to wait for a message at `now`: no longer than until the
    /// next heartbeat or the next expiry is due.
    fn wait(&self, now: Instant) -> Duration {
        let beat = (self.beat + DEFAULT_HEARTBEAT).saturating_duration_since(now);
        let expiry = (self.expired + EXPIRE_INTERVAL).saturating_duration_since(now);
        beat.min(expiry)
    }

    /// One pass of the read loop at `now`, after `msg` was read (`None`:
    /// nothing arrived in time, or what arrived was not an FRP message).
    /// Returns what to do, in order.
    fn read(&mut self, msg: Option<&FrpMessage>, now: Instant) -> Vec<ReadAction> {
        let mut actions = Vec::new();
        match msg {
            Some(msg) if matches!(msg.body, FrpBody::BusSync(_)) => {
                actions.push(ReadAction::Announce);
            }
            Some(msg) => {
                let arrival = now.saturating_duration_since(self.start).as_secs_f64();
                if let Some(shot) = self.intake.ingest_at(msg, now) {
                    // The ready states first, so a shot's own state change
                    // is on the app before the shot.
                    self.report_lm_states(&mut actions);
                    eprintln!(
                        "Shot #{} from {} arrived at {arrival:.3}s",
                        shot.shot_number, shot.device
                    );
                    actions.push(ReadAction::Report(IntakeEvent::Shot(ArrivedShot {
                        shot,
                        arrival,
                    })));
                }
            }
            None => {}
        }
        if now.saturating_duration_since(self.beat) >= DEFAULT_HEARTBEAT {
            self.beat = now;
            actions.push(ReadAction::Announce);
        }
        if now.saturating_duration_since(self.expired) >= EXPIRE_INTERVAL {
            self.expired = now;
            self.intake.expire_at(now);
        }
        self.report_lm_states(&mut actions);
        actions
    }

    /// Report the ready states if they differ from what the app has.
    fn report_lm_states(&mut self, actions: &mut Vec<ReadAction>) {
        if self.intake.lm_states != self.reported {
            self.reported.clone_from(&self.intake.lm_states);
            actions.push(ReadAction::Report(IntakeEvent::LmStates(
                self.reported.clone(),
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// The actor id of the test's launch monitor.
    const LM: &str = "7d2f9a41c03be856";

    /// A message from the test's launch monitor.
    fn lm(body: impl Into<flightrelay::FrpBody>) -> FrpMessage {
        FrpMessage::new(LM, body)
    }

    /// The test launch monitor's `bus.actor`, named `name`, with `blockers`.
    fn lm_actor(name: &str, blockers: &[&str]) -> BusActor {
        let mut actor = BusActor::new(name, 2.0).with_role(roles::LAUNCH_MONITOR);
        actor.blockers = blockers.iter().map(|&b| b.to_owned()).collect();
        actor
    }

    /// The lifecycle messages of shot `n`: trigger, launch, finished.
    fn shot_messages(n: u32, speed_mph: f64, elevation_deg: f64) -> Vec<FrpMessage> {
        let key = flightrelay::ShotKey {
            shot_id: format!("shot-{n}"),
            shot_number: n,
        };
        let launch = flightrelay::Launch {
            launch_speed: Some(flightrelay::Velocity::MilesPerHour(speed_mph)),
            launch_elevation: Some(elevation_deg),
            launch_azimuth: Some(0.0),
            backspin_rpm: Some(3000),
            sidespin_rpm: Some(0),
            ..flightrelay::Launch::default()
        };
        vec![
            lm(flightrelay::ShotTrigger::new(key.clone())),
            lm(flightrelay::ShotLaunch {
                key: key.clone(),
                launch,
            }),
            lm(flightrelay::ShotFinished { key }),
        ]
    }

    #[test]
    fn frp_url_comes_from_the_flag() {
        let args = |a: &[&str]| a.iter().map(|&s| s.to_owned()).collect::<Vec<_>>();
        assert_eq!(
            frp_url(&args(&["cyberflight"])).expect("default"),
            DEFAULT_FRP_URL
        );
        assert_eq!(
            frp_url(&args(&[
                "cyberflight",
                "--stream",
                "--frp",
                "ws://pc:5880/frp"
            ]))
            .expect("ws"),
            "ws://pc:5880/frp"
        );
        assert_eq!(
            frp_url(&args(&["cyberflight", "--frp", "wss://pc/frp"])).expect("wss"),
            "wss://pc/frp"
        );
    }

    #[test]
    fn frp_url_must_be_a_websocket_url() {
        let args = |a: &[&str]| a.iter().map(|&s| s.to_owned()).collect::<Vec<_>>();
        let missing = frp_url(&args(&["cyberflight", "--frp"])).unwrap_err();
        assert!(missing.to_string().contains("needs a URL"), "{missing}");
        for bad in ["--stream", "pc:5880", "http://pc:5880/frp"] {
            let err = frp_url(&args(&["cyberflight", "--frp", bad])).unwrap_err();
            assert!(err.to_string().contains("ws://"), "{bad}: {err}");
        }
    }

    // Every port an intake in these tests dials stays bound by the test's
    // device endpoint until that intake can dial no more. A released port
    // can be bound again by anything on the machine, such as another test's
    // device, whose connection a stray dial would then take.

    /// A device listening on a free local port, and the URL to dial it.
    fn device() -> (FrpEndpoint, String) {
        let device = FrpEndpoint::open(EndpointConfig::new(Transport::listen("127.0.0.1:0")))
            .expect("device endpoint");
        let addr = device.local_addr().expect("addr").expect("bound");
        (device, format!("ws://{addr}/frp"))
    }

    /// Cyberflight's actor id in these tests.
    const ME: &str = "c19e04b7a2d5f318";

    /// Read cyberflight's greeting on `conn`: its `bus.actor`, then
    /// `bus.sync`.
    fn greeted(conn: &mut FrpConnection) {
        let actor = conn.recv().expect("bus.actor");
        assert_eq!(actor, actor_message(ME));
        let sync = conn.recv().expect("bus.sync");
        assert_eq!(
            (sync.from.as_str(), sync.kind()),
            (ME, "bus.sync"),
            "{sync:?}"
        );
    }

    /// The longest any wait in these tests lasts: a guard against a hang,
    /// never a quiet window.
    const PATIENCE: Duration = Duration::from_secs(5);

    /// Quick redials, so the socket test runs in milliseconds.
    const QUICK: Duration = Duration::from_millis(50);

    /// Wait at most [`PATIENCE`] for `thread` to end and return its result.
    /// Fails the test, naming `what`, if it is still running, and with the
    /// thread's own message if it panicked.
    fn join<T: Send + 'static>(thread: thread::JoinHandle<T>, what: &str) -> T {
        let (done_tx, done) = mpsc::channel();
        thread::spawn(move || {
            let _ = done_tx.send(thread.join());
        });
        done.recv_timeout(PATIENCE)
            .unwrap_or_else(|_| panic!("{what} is still running after {PATIENCE:?}"))
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }

    /// Receive intake events until one is `last`, for at most [`PATIENCE`]
    /// in all. Returns them, `last` included.
    fn recv_until(
        events: &mpsc::Receiver<IntakeEvent>,
        last: impl Fn(&IntakeEvent) -> bool,
    ) -> Vec<IntakeEvent> {
        let deadline = Instant::now() + PATIENCE;
        let mut seen = Vec::new();
        loop {
            match events.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(event) => {
                    let done = last(&event);
                    seen.push(event);
                    if done {
                        return seen;
                    }
                }
                Err(e) => panic!("{e} after {PATIENCE:?}; received {:?}", labels(&seen)),
            }
        }
    }

    /// The intake events as text, without dial starts.
    fn labels(events: &[IntakeEvent]) -> Vec<String> {
        events
            .iter()
            .filter(|e| !matches!(e, IntakeEvent::Connecting))
            .map(|e| match e {
                IntakeEvent::Shot(s) => format!("shot {}", s.shot.shot_number),
                IntakeEvent::LmStates(states) => format!("ready {states:?}"),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn the_default_device_is_dialled_over_ipv4() {
        assert!(DEFAULT_FRP_URL.starts_with("ws://127.0.0.1:"));
        assert!(DEFAULT_FRP_URL.ends_with(&format!(
            ":{}{}",
            flightrelay::DEFAULT_PORT,
            flightrelay::DEFAULT_PATH
        )));
    }

    #[test]
    fn the_intake_dials_every_reconnect_interval() {
        let t0 = Instant::now();
        let ms = |ms: u64| t0 + Duration::from_millis(ms);
        let mut dialer = Dialer::new(RECONNECT_INTERVAL);
        assert_eq!(dialer.step(t0), DialStep::Dial);
        // Refused after 200 ms: the interval runs from the start of the dial.
        assert_eq!(
            dialer.step(ms(200)),
            DialStep::Wait(Duration::from_millis(4_800))
        );
        assert_eq!(
            dialer.step(ms(4_999)),
            DialStep::Wait(Duration::from_millis(1))
        );
        assert_eq!(dialer.step(ms(5_000)), DialStep::Dial);
        // A connection lost after the interval is redialled at once.
        assert_eq!(dialer.step(ms(60_000)), DialStep::Dial);
    }

    /// The events in `actions`, with each `bus.actor` sent as `"announce"`.
    fn read_labels(actions: Vec<ReadAction>) -> Vec<String> {
        actions
            .into_iter()
            .map(|action| match action {
                ReadAction::Announce => "announce".to_owned(),
                ReadAction::Report(event) => labels(&[event]).concat(),
            })
            .collect()
    }

    /// Read `msgs` at `now`; returns the shots reported.
    fn read_shots(
        reader: &mut IntakeReader,
        msgs: &[FrpMessage],
        now: Instant,
    ) -> Vec<ArrivedShot> {
        msgs.iter()
            .flat_map(|msg| reader.read(Some(msg), now))
            .filter_map(|action| match action {
                ReadAction::Report(IntakeEvent::Shot(shot)) => Some(shot),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn shots_keep_their_arrival_times() {
        let start = Instant::now();
        let connected = start + Duration::from_secs(1);
        let mut reader = IntakeReader::new(start, connected);
        let (tx, rx) = mpsc::channel();
        let mut state = FrpState::from_events(rx);
        for (n, read) in [(1, 1.1), (2, 1.4)] {
            let at = start + Duration::from_secs_f64(read);
            for shot in read_shots(&mut reader, &shot_messages(n, 140.0, 14.0), at) {
                tx.send(IntakeEvent::Shot(shot)).expect("send");
            }
        }

        // The app polls later: the shots keep the times they were read.
        let shots = state.poll();
        let got: Vec<(u32, f64)> = shots
            .iter()
            .map(|s| (s.shot.shot_number, s.arrival))
            .collect();
        assert_eq!(got, [(1, 1.1), (2, 1.4)]);
    }

    #[test]
    fn the_intake_answers_bus_sync_with_its_actor() {
        let start = Instant::now();
        let mut reader = IntakeReader::new(start, start);
        let sync = FrpMessage::new(LM, flightrelay::BusSync::default());
        assert_eq!(read_labels(reader.read(Some(&sync), start)), ["announce"]);
    }

    #[test]
    fn the_intake_republishes_its_actor_every_heartbeat() {
        let start = Instant::now();
        let mut reader = IntakeReader::new(start, start);
        assert_eq!(reader.wait(start), EXPIRE_INTERVAL.min(DEFAULT_HEARTBEAT));
        let early = start + DEFAULT_HEARTBEAT - Duration::from_millis(1);
        assert!(reader.read(None, early).is_empty());
        assert_eq!(reader.wait(early), Duration::from_millis(1));
        let due = start + DEFAULT_HEARTBEAT;
        assert_eq!(read_labels(reader.read(None, due)), ["announce"]);
        assert!(reader.read(None, due).is_empty());
        assert_eq!(
            read_labels(reader.read(None, due + DEFAULT_HEARTBEAT)),
            ["announce"]
        );
    }

    #[test]
    fn a_silent_launch_monitor_is_forgotten() {
        let start = Instant::now();
        let mut reader = IntakeReader::new(start, start);
        // Heartbeat 2 s: it is forgotten once 6 s pass without its `bus.actor`.
        let actor = lm(lm_actor("Eagle One", &[]));
        assert_eq!(
            read_labels(reader.read(Some(&actor), start)),
            [r#"ready {"Eagle One": true}"#]
        );
        let at = |s: f64| start + Duration::from_secs_f64(s);
        assert!(
            !read_labels(reader.read(None, at(5.5)))
                .iter()
                .any(|l| l.starts_with("ready"))
        );
        let gone = read_labels(reader.read(None, at(6.5)));
        assert_eq!(
            gone.last().map(String::as_str),
            Some("ready {}"),
            "{gone:?}"
        );
    }

    // The intake thread around `Dialer` and `IntakeReader`, on real sockets:
    // it connects and greets, reads, redials when the peer closes, and ends
    // at its next report once the app is gone.
    #[test]
    fn the_intake_reads_redials_and_ends() {
        let (mut device, url) = device();
        let server = thread::spawn(move || {
            let mut conn = device.establish().expect("handshake");
            greeted(&mut conn);
            conn.send(&lm(lm_actor("Eagle One", &[]))).expect("send");
            // Shot 3 is cut off by the close; its end comes on the next connection.
            let shot3 = shot_messages(3, 140.0, 14.0);
            for msg in &shot3[..2] {
                conn.send(msg).expect("send");
            }
            drop(conn);

            let mut conn = device.establish().expect("second handshake");
            greeted(&mut conn);
            conn.send(&shot3[2]).expect("send");
            for msg in shot_messages(4, 140.0, 14.0) {
                conn.send(&msg).expect("send");
            }
            (device, conn)
        });

        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&wakes);
        let wake = move || {
            counter.fetch_add(1, Ordering::SeqCst);
        };
        let (events, intake) = spawn_intake(&url, ME, Instant::now(), wake, QUICK).expect("intake");
        let seen = recv_until(&events, |e| matches!(e, IntakeEvent::Shot(_)));
        let (_device, mut conn) = join(server, "the device (it accepts twice)");
        assert_eq!(
            labels(&seen),
            [
                "Connected",
                r#"ready {"Eagle One": true}"#,
                "Disconnected",
                "Connected",
                // Shot 3 was cut off by the close and is forgotten.
                "shot 4",
            ]
        );

        // The app goes; the intake ends at the next shot.
        drop(events);
        for msg in shot_messages(5, 140.0, 14.0) {
            conn.send(&msg).expect("send");
        }
        join(intake, "the intake");
        // One wake per event the app received; none for a dial start.
        assert_eq!(wakes.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn the_app_forgets_ready_states_with_the_connection() {
        let (tx, rx) = mpsc::channel();
        let mut state = FrpState::from_events(rx);
        let ready = HashMap::from([("Eagle One".to_owned(), true)]);
        for event in [IntakeEvent::Connected, IntakeEvent::LmStates(ready.clone())] {
            tx.send(event).expect("send");
        }
        assert!(state.poll().is_empty());
        assert!(state.connected);
        assert_eq!(state.lm_states, ready);
        tx.send(IntakeEvent::Disconnected).expect("send");
        assert!(state.poll().is_empty());
        assert!(!state.connected);
        assert!(state.lm_states.is_empty());
    }

    #[test]
    fn a_stopped_intake_is_not_connected() {
        let (tx, rx) = mpsc::channel();
        let mut state = FrpState::from_events(rx);
        tx.send(IntakeEvent::Connected).expect("send");
        drop(tx);
        assert!(state.poll().is_empty());
        assert!(!state.connected);
        assert!(state.stopped);
    }

    #[test]
    fn shot_without_speed_is_not_taken_in() {
        let mut intake = ShotIntake::default();
        let mut msgs = shot_messages(1, 160.0, 11.0);
        if let flightrelay::FrpBody::ShotLaunch(m) = &mut msgs[1].body {
            m.launch.launch_speed = None;
        }
        let now = Instant::now();
        assert!(msgs.iter().all(|msg| intake.ingest_at(msg, now).is_none()));
    }

    #[test]
    fn launch_monitors_are_ready_without_blockers() {
        let now = Instant::now();
        let mut intake = ShotIntake::default();
        assert!(intake.lm_states.is_empty());
        intake.ingest_at(&lm(lm_actor("Eagle One", &["not_armed"])), now);
        assert_eq!(
            intake.lm_states,
            HashMap::from([("Eagle One".to_owned(), false)])
        );
        intake.ingest_at(&lm(lm_actor("Eagle One", &[])), now);
        assert_eq!(
            intake.lm_states,
            HashMap::from([("Eagle One".to_owned(), true)])
        );
        // Another actor with no launch monitor role is not listed; a second
        // launch monitor with the same name is told apart by its id.
        intake.ingest_at(
            &FrpMessage::new(
                "aaaa000000000000",
                BusActor::new("Shot Log", 2.0).with_role(roles::COLLECTOR),
            ),
            now,
        );
        intake.ingest_at(
            &FrpMessage::new("bbbb000000000000", lm_actor("Eagle One", &["no_ball"])),
            now,
        );
        assert_eq!(
            intake.lm_states,
            HashMap::from([
                (format!("Eagle One ({LM})"), true),
                ("Eagle One (bbbb000000000000)".to_owned(), false),
            ])
        );
        // An actor that leaves is forgotten.
        intake.ingest_at(
            &FrpMessage::new(
                "bbbb000000000000",
                flightrelay::Removal::of(flightrelay::StateKind::BusActor),
            ),
            now,
        );
        assert_eq!(
            intake.lm_states,
            HashMap::from([("Eagle One".to_owned(), true)])
        );
    }

    #[test]
    fn a_late_flight_does_not_take_the_shot_in_again() {
        let mut intake = ShotIntake::default();
        let now = Instant::now();
        let shots = shot_messages(1, 160.0, 11.0)
            .iter()
            .filter_map(|msg| intake.ingest_at(msg, now))
            .count();
        assert_eq!(shots, 1);
        let late = FrpMessage::new(
            "e4a7136b9f02cd58",
            flightrelay::ShotFlight {
                key: flightrelay::ShotKey::new("shot-1", 1),
                flight: flightrelay::FlightResult::new(flightrelay::Basis::Modeled),
            },
        );
        assert!(intake.ingest_at(&late, now).is_none());
    }

    #[test]
    fn cyberflight_announces_itself_as_a_collector() {
        let me = actor_message(ME);
        assert_eq!(me.from, ME);
        let flightrelay::FrpBody::BusActor(actor) = &me.body else {
            panic!("{me:?}");
        };
        assert_eq!(actor.name, ACTOR_NAME);
        assert!(actor.has_role(roles::COLLECTOR));
        assert!(actor.is_clear());
        assert!((actor.heartbeat - DEFAULT_HEARTBEAT.as_secs_f64()).abs() < 1e-9);
    }
}
