//! `cyberflight.log`: a copy of everything cyberflight prints to stderr,
//! next to the executable, each line stamped with the local wall-clock time
//! (`HH:MM:SS.mmm`).
//!
//! [`start`] points the process's stderr at a pipe. A thread reads the pipe,
//! writes each line to the log file with its time and hands it on, as it
//! came, to a second thread that writes the original stderr. Up to
//! [`CONSOLE_BACKLOG`] lines wait for a console that stops reading (Windows
//! stops reading while text is selected in the console); beyond that, lines
//! are left out of the console, never out of the log, and printing never
//! waits for the console.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;

/// The log's file name, in the executable's directory.
pub const LOG_FILE_NAME: &str = "cyberflight.log";

/// Lines that wait for a console that is not reading.
const CONSOLE_BACKLOG: usize = 1024;

/// Longest line kept whole; longer ones are split.
const MAX_LINE: usize = 64 * 1024;

/// How long the end of the output may take to reach the log at exit.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);

/// The log, open until dropped: dropping it points stderr back where it was
/// and waits (at most [`FLUSH_TIMEOUT`]) for the last lines to be written.
pub struct Log {
    redirect: Option<sys::Redirect>,
    /// Closed when the log thread has written everything.
    done: mpsc::Receiver<()>,
}

impl Drop for Log {
    fn drop(&mut self) {
        if let Some(redirect) = self.redirect.take() {
            redirect.restore();
        }
        // The pipe's last write end is closed: the log thread reads what is
        // left, then ends.
        let _ = self.done.recv_timeout(FLUSH_TIMEOUT);
    }
}

/// Start mirroring stderr into [`LOG_FILE_NAME`], truncating it. If that
/// fails, says so once on stderr and returns `None`; output then goes to
/// stderr only.
pub fn start() -> Option<Log> {
    match open() {
        Ok((log, path)) => {
            eprintln!("Logging to {}", path.display());
            Some(log)
        }
        Err(e) => {
            eprintln!("WARN: not writing {LOG_FILE_NAME}: {e:#}");
            None
        }
    }
}

fn open() -> anyhow::Result<(Log, PathBuf)> {
    let exe = std::env::current_exe().context("cannot find the executable")?;
    let path = exe.with_file_name(LOG_FILE_NAME);
    let mut file =
        File::create(&path).with_context(|| format!("cannot create {}", path.display()))?;
    file.write_all(header(sys::local_time()).as_bytes())
        .with_context(|| format!("cannot write {}", path.display()))?;
    let (reader, writer) = sys::pipe().context("cannot create a pipe")?;
    // Without a console (no stderr to copy) the lines go to the log only.
    let console = sys::stderr_copy().ok();

    let (done_tx, done) = mpsc::channel::<()>();
    thread::Builder::new()
        .name("log".to_owned())
        .spawn(move || {
            run_log(reader, file, console);
            drop(done_tx);
        })
        .context("cannot start the log thread")?;

    // The thread is reading before stderr is pointed at the pipe; on failure
    // the write end is dropped and the thread ends.
    let redirect = sys::Redirect::install(writer).context("cannot redirect stderr")?;
    Ok((
        Log {
            redirect: Some(redirect),
            done,
        },
        path,
    ))
}

/// The log thread: copy `pipe` into `file` and onto `console` until every
/// write end of the pipe is closed.
fn run_log(pipe: File, mut file: File, console: Option<File>) {
    let (lines, backlog) = mpsc::sync_channel::<Vec<u8>>(CONSOLE_BACKLOG);
    let console = console.and_then(|mut out| {
        thread::Builder::new()
            .name("log-console".to_owned())
            .spawn(move || {
                for line in backlog {
                    // Nowhere to report a console that cannot be written.
                    let _ = out.write_all(&line);
                }
            })
            .ok()
    });
    let mut skipped = 0usize;
    tee(pipe, &mut file, stamp, |line| {
        if console.is_none() {
            return;
        }
        if skipped > 0 {
            let note = format!("[{skipped} lines not shown here; see {LOG_FILE_NAME}]\n");
            if lines.try_send(note.into_bytes()).is_err() {
                skipped += 1;
                return;
            }
            skipped = 0;
        }
        if lines.try_send(line.to_vec()).is_err() {
            skipped += 1;
        }
    });
    drop(lines);
    // Bounded by `Log::drop`'s wait when the console is not reading.
    if let Some(console) = console {
        let _ = console.join();
    }
}

/// Copy `input` line by line: to `log` with `stamp()` and a space in
/// front, and as it came to `console`. A line still open when `input` ends
/// is passed on, with a newline added in the log.
fn tee(
    mut input: impl Read,
    log: &mut impl Write,
    stamp: impl Fn() -> String,
    mut console: impl FnMut(&[u8]),
) {
    let mut line = Vec::new();
    let mut emit = |line: &mut Vec<u8>| {
        let mut stamped = stamp().into_bytes();
        stamped.push(b' ');
        stamped.extend_from_slice(line);
        if stamped.last() != Some(&b'\n') {
            stamped.push(b'\n');
        }
        // A log that cannot be written (disk full) still leaves the console.
        let _ = log.write_all(&stamped);
        console(line);
        line.clear();
    };
    let mut buf = [0u8; 4096];
    loop {
        let n = match input.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        for &byte in &buf[..n] {
            line.push(byte);
            if byte == b'\n' || line.len() >= MAX_LINE {
                emit(&mut line);
            }
        }
    }
    if !line.is_empty() {
        emit(&mut line);
    }
}

/// A local wall-clock time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalTime {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub millis: u16,
}

/// The time of day a line is stamped with: local `HH:MM:SS.mmm`, or UTC
/// with a `Z` when the local time is not available.
fn stamp() -> String {
    match sys::local_time() {
        Some(t) => format_time_of_day(t),
        None => utc_time_of_day(SystemTime::now()),
    }
}

fn format_time_of_day(t: LocalTime) -> String {
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        t.hour, t.minute, t.second, t.millis
    )
}

/// UTC `HH:MM:SS.mmmZ` of `time`.
fn utc_time_of_day(time: SystemTime) -> String {
    let since_epoch = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since_epoch.as_secs() % 86_400;
    format!(
        "{:02}:{:02}:{:02}.{:03}Z",
        secs / 3600,
        secs / 60 % 60,
        secs % 60,
        since_epoch.subsec_millis()
    )
}

/// The log's first line: what wrote it, when, and the clock of the stamps.
fn header(now: Option<LocalTime>) -> String {
    let version = env!("CARGO_PKG_VERSION");
    match now {
        Some(t) => format!(
            "cyberflight {version} log, started {:04}-{:02}-{:02} {:02}:{:02}:{:02}; \
             lines are stamped with the local time\n",
            t.year, t.month, t.day, t.hour, t.minute, t.second
        ),
        None => format!(
            "cyberflight {version} log; the local time is not available, \
             lines are stamped with the UTC time (Z)\n"
        ),
    }
}

#[cfg(unix)]
mod sys {
    use std::fs::File;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, RawFd};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::LocalTime;

    /// The standard error file descriptor.
    const STDERR: RawFd = 2;

    fn check(result: libc::c_int) -> io::Result<libc::c_int> {
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(result)
        }
    }

    /// Take ownership of `fd`, a descriptor just returned by the OS.
    fn own(fd: RawFd) -> File {
        // SAFETY: `fd` was just created by `pipe` or `fcntl` and nothing else
        // owns it.
        unsafe { File::from_raw_fd(fd) }
    }

    /// A pipe, `(read end, write end)`, both closed on exec so child
    /// processes do not hold it open.
    pub fn pipe() -> io::Result<(File, File)> {
        let mut fds = [0; 2];
        // SAFETY: `fds` has room for the two descriptors `pipe` writes.
        check(unsafe { libc::pipe(fds.as_mut_ptr()) })?;
        let (read, write) = (own(fds[0]), own(fds[1]));
        for end in [&read, &write] {
            // SAFETY: `end` is an open descriptor; F_SETFD takes an int flag.
            check(unsafe { libc::fcntl(end.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) })?;
        }
        Ok((read, write))
    }

    /// A copy of the current stderr, closed on exec.
    pub fn stderr_copy() -> io::Result<File> {
        // SAFETY: F_DUPFD_CLOEXEC duplicates an open descriptor (stderr) to
        // the lowest free one at or above the int argument.
        let fd = check(unsafe { libc::fcntl(STDERR, libc::F_DUPFD_CLOEXEC, 0) })?;
        Ok(own(fd))
    }

    /// Make `fd` the process's stderr.
    fn set_stderr(fd: &File) -> io::Result<()> {
        // SAFETY: both descriptors are open; `dup2` replaces descriptor 2.
        check(unsafe { libc::dup2(fd.as_raw_fd(), STDERR) })?;
        Ok(())
    }

    /// Stderr pointed at a pipe, and where it pointed before.
    pub struct Redirect {
        original: File,
        writer: File,
    }

    impl Redirect {
        /// Point stderr at `writer`.
        pub fn install(writer: File) -> io::Result<Self> {
            let original = stderr_copy()?;
            set_stderr(&writer)?;
            Ok(Self { original, writer })
        }

        /// Point stderr back where it was and close the pipe's write ends.
        pub fn restore(self) {
            // On failure stderr stays on the pipe, which the log thread keeps
            // reading until the process ends.
            let _ = set_stderr(&self.original);
            drop(self.writer);
        }
    }

    pub fn local_time() -> Option<LocalTime> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
        let secs = libc::time_t::try_from(now.as_secs()).ok()?;
        // SAFETY: `tm` is plain integers and a pointer; all-zero is valid.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        // SAFETY: both pointers are valid for the call; `localtime_r` writes
        // only `tm`.
        if unsafe { libc::localtime_r(&secs, &mut tm) }.is_null() {
            return None;
        }
        Some(LocalTime {
            year: u16::try_from(tm.tm_year + 1900).ok()?,
            month: u8::try_from(tm.tm_mon + 1).ok()?,
            day: u8::try_from(tm.tm_mday).ok()?,
            hour: u8::try_from(tm.tm_hour).ok()?,
            minute: u8::try_from(tm.tm_min).ok()?,
            second: u8::try_from(tm.tm_sec).ok()?,
            millis: u16::try_from(now.subsec_millis()).ok()?,
        })
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, RawHandle};
    use std::ptr;

    use super::LocalTime;

    /// `STD_ERROR_HANDLE`: `(DWORD)-12`.
    const STD_ERROR_HANDLE: u32 = 0xFFFF_FFF4;
    /// `DUPLICATE_SAME_ACCESS`.
    const DUPLICATE_SAME_ACCESS: u32 = 2;

    /// `SYSTEMTIME`.
    #[repr(C)]
    #[derive(Default)]
    struct SystemTime {
        year: u16,
        month: u16,
        day_of_week: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        milliseconds: u16,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(std_handle: u32) -> RawHandle;
        fn SetStdHandle(std_handle: u32, handle: RawHandle) -> i32;
        fn CreatePipe(
            read: *mut RawHandle,
            write: *mut RawHandle,
            attributes: *const c_void,
            size: u32,
        ) -> i32;
        fn GetCurrentProcess() -> RawHandle;
        fn DuplicateHandle(
            source_process: RawHandle,
            source: RawHandle,
            target_process: RawHandle,
            target: *mut RawHandle,
            access: u32,
            inherit: i32,
            options: u32,
        ) -> i32;
        fn GetLocalTime(time: *mut SystemTime);
    }

    fn check(ok: i32) -> io::Result<()> {
        if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// A pipe, `(read end, write end)`. Its handles are not inherited by
    /// child processes.
    pub fn pipe() -> io::Result<(File, File)> {
        let mut read = ptr::null_mut();
        let mut write = ptr::null_mut();
        // SAFETY: both out pointers are valid; null attributes make the
        // handles non-inheritable; size 0 is the default buffer.
        check(unsafe { CreatePipe(&mut read, &mut write, ptr::null(), 0) })?;
        // SAFETY: `CreatePipe` succeeded, so both handles are open and owned
        // by nothing else.
        Ok(unsafe { (File::from_raw_handle(read), File::from_raw_handle(write)) })
    }

    /// The process's current stderr handle (null or invalid without one).
    fn stderr_handle() -> RawHandle {
        // SAFETY: `GetStdHandle` only reads the process's standard handles.
        unsafe { GetStdHandle(STD_ERROR_HANDLE) }
    }

    /// An owned copy of the current stderr handle.
    pub fn stderr_copy() -> io::Result<File> {
        let original = stderr_handle();
        if original.is_null() || original as isize == -1 {
            return Err(io::Error::other("no stderr"));
        }
        let mut copy = ptr::null_mut();
        // SAFETY: `original` is the process's stderr handle and the pseudo
        // handle of the current process is valid in both process arguments;
        // `copy` is a valid out pointer.
        check(unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                original,
                GetCurrentProcess(),
                &mut copy,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        })?;
        // SAFETY: `DuplicateHandle` succeeded, so `copy` is a new handle this
        // function owns.
        Ok(unsafe { File::from_raw_handle(copy) })
    }

    /// Stderr pointed at a pipe, and where it pointed before.
    pub struct Redirect {
        /// Not owned: the handle the process started with.
        original: RawHandle,
        writer: File,
    }

    impl Redirect {
        /// Point stderr at `writer`. Rust's stderr looks the handle up on
        /// every write, so this takes effect at once.
        pub fn install(writer: File) -> io::Result<Self> {
            let original = stderr_handle();
            // SAFETY: `writer` is an open handle, kept open until `restore`.
            check(unsafe { SetStdHandle(STD_ERROR_HANDLE, writer.as_raw_handle()) })?;
            Ok(Self { original, writer })
        }

        /// Point stderr back where it was and close the pipe's write end.
        pub fn restore(self) {
            // SAFETY: `original` is the handle stderr had before `install`,
            // never closed by this module.
            let restored = check(unsafe { SetStdHandle(STD_ERROR_HANDLE, self.original) });
            if restored.is_ok() {
                drop(self.writer);
            } else {
                // Stderr still points at the pipe: keep its write end open.
                std::mem::forget(self.writer);
            }
        }
    }

    pub fn local_time() -> Option<LocalTime> {
        let mut t = SystemTime::default();
        // SAFETY: `t` is a valid `SYSTEMTIME` to write.
        unsafe { GetLocalTime(&mut t) };
        Some(LocalTime {
            year: t.year,
            month: u8::try_from(t.month).ok()?,
            day: u8::try_from(t.day).ok()?,
            hour: u8::try_from(t.hour).ok()?,
            minute: u8::try_from(t.minute).ok()?,
            second: u8::try_from(t.second).ok()?,
            millis: t.milliseconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tee_all(input: &[u8]) -> (String, Vec<String>) {
        let mut log = Vec::new();
        let mut console = Vec::new();
        tee(
            input,
            &mut log,
            || "12:34:56.789".to_owned(),
            |line| console.push(String::from_utf8_lossy(line).into_owned()),
        );
        (String::from_utf8(log).expect("utf-8"), console)
    }

    #[test]
    fn each_line_is_stamped_in_the_log_only() {
        let (log, console) = tee_all(b"Connecting...\nConnected (FRP 0.1.0)\n");
        assert_eq!(
            log,
            "12:34:56.789 Connecting...\n12:34:56.789 Connected (FRP 0.1.0)\n"
        );
        assert_eq!(console, ["Connecting...\n", "Connected (FRP 0.1.0)\n"]);
    }

    #[test]
    fn an_open_last_line_is_passed_on() {
        let (log, console) = tee_all(b"one\ntwo");
        assert_eq!(log, "12:34:56.789 one\n12:34:56.789 two\n");
        assert_eq!(console, ["one\n", "two"]);
    }

    #[test]
    fn a_line_split_across_reads_stays_one_line() {
        // One byte per read.
        struct Trickle<'a>(&'a [u8]);
        impl Read for Trickle<'_> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let Some((&first, rest)) = self.0.split_first() else {
                    return Ok(0);
                };
                buf[0] = first;
                self.0 = rest;
                Ok(1)
            }
        }
        let mut log = Vec::new();
        tee(Trickle(b"Shot #1\n"), &mut log, || "T".to_owned(), |_| {});
        assert_eq!(log, b"T Shot #1\n");
    }

    #[test]
    fn a_very_long_line_is_split() {
        let input = vec![b'x'; MAX_LINE + 10];
        let (_, console) = tee_all(&input);
        assert_eq!(console.len(), 2);
        assert_eq!(console[0].len(), MAX_LINE);
    }

    #[test]
    fn the_stamp_is_the_time_of_day_to_the_millisecond() {
        let t = LocalTime {
            year: 2026,
            month: 10,
            day: 28,
            hour: 7,
            minute: 5,
            second: 9,
            millis: 42,
        };
        assert_eq!(format_time_of_day(t), "07:05:09.042");
        assert!(header(Some(t)).contains("started 2026-10-28 07:05:09"));
        assert!(header(Some(t)).contains("local time"));
        assert!(header(None).contains("UTC"));
        let time = UNIX_EPOCH + Duration::from_millis(86_400_000 * 3 + 3_723_004);
        assert_eq!(utc_time_of_day(time), "01:02:03.004Z");
    }

    #[test]
    fn the_local_time_is_available() {
        let t = sys::local_time().expect("local time");
        assert!(t.year >= 2024);
        assert!((1..=12).contains(&t.month));
        assert!(t.hour < 24 && t.minute < 60 && t.second <= 60 && t.millis < 1000);
    }
}
