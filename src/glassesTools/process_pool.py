"""Process pool with job scheduling, prioritization, and progress tracking."""

import dataclasses
import enum
import multiprocessing
import multiprocessing.managers
import threading
import time
import types
import typing

import pebble

from . import json, utils

ProcessFuture: typing.TypeAlias = pebble.ProcessFuture
_UserDataT = typing.TypeVar("_UserDataT")


class CounterContext:
    """Context manager that increments a counter on each entry."""

    _count = -1  # so that first number is 0

    def __enter__(self) -> None:
        """Increment the internal counter."""
        self._count += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the context (no-op)."""

    def get_count(self) -> int:
        """Return the current counter value."""
        return self._count


class State(enum.IntEnum):
    """Task lifecycle states, storable in JSON files."""

    Not_Run = enum.auto()
    Pending = enum.auto()
    Running = enum.auto()
    Completed = enum.auto()
    Canceled = enum.auto()
    Failed = enum.auto()

    @property
    def displayable_name(self) -> str:
        """Return a human-readable name with underscores replaced by spaces."""
        return self.name.replace("_", " ")


json.register_type(
    json.TypeEntry(State, "__enum.process.State__", utils.enum_val_2_str, lambda x: getattr(State, x.split(".")[1]))
)


class ProcessWaiter:
    """Routes completion through to user callback."""

    def __init__(
        self,
        job_id: int,
        user_data: _UserDataT,
        done_callback: typing.Callable[[ProcessFuture, _UserDataT, int, State], None],
    ) -> None:
        """Store the callback and job metadata."""
        self.done_callback = done_callback
        self.job_id = job_id
        self.user_data = user_data

    def add_result(self, future: ProcessFuture) -> None:
        """Handle a successful future result."""
        self._notify(future, State.Completed)

    def add_exception(self, future: ProcessFuture) -> None:
        """Handle a future that raised an exception."""
        self._notify(future, State.Failed)

    def add_cancelled(self, future: ProcessFuture) -> None:
        """Handle a cancelled future."""
        self._notify(future, State.Canceled)

    def _notify(self, future: ProcessFuture, state: State) -> None:
        self.done_callback(future, self.job_id, self.user_data, state)


class PoolJob(typing.NamedTuple):
    """A scheduled pool job bundling a future with user data."""

    future: ProcessFuture
    user_data: _UserDataT


class ProcessPool:
    """Wrapper around pebble.ProcessPool with job tracking and callbacks."""

    def __init__(self, num_workers: int = 2) -> None:
        """Initialize the pool with a target number of workers."""
        self.num_workers = num_workers
        self.auto_cleanup_if_no_work = False

        # NB: pool is only started in run() once needed
        self._pool: pebble.pool.ProcessPool = None
        self._jobs: dict[int, PoolJob] = None
        self._job_id_provider: CounterContext = CounterContext()
        self._lock: threading.Lock = threading.Lock()

    def _cleanup(self) -> None:
        # cancel all pending and running jobs
        self.cancel_all_jobs()

        # stop pool
        if self._pool and self._pool.active:
            self._pool.stop()
            self._pool.join()
        self._pool = None
        self._jobs = None

    def cleanup(self) -> None:
        """Cancel all jobs and shut down the pool."""
        with self._lock:
            self._cleanup()

    def cleanup_if_no_jobs(self) -> None:
        """Shut down the pool only if there are no remaining jobs."""
        with self._lock:
            self._cleanup_if_no_jobs()

    def _cleanup_if_no_jobs(self) -> None:
        # NB: lock must be acquired when calling this
        if self._pool and not self._jobs:
            self._cleanup()

    def set_num_workers(self, num_workers: int) -> None:
        """Set the worker count (takes effect on next pool restart)."""
        # NB: doesn't change number of workers on an active pool, only takes effect when pool is restarted
        self.num_workers = num_workers

    def run(
        self,
        fn: typing.Callable,
        user_data: _UserDataT = None,
        done_callback: typing.Callable[[ProcessFuture, _UserDataT, int, State], None] | None = None,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> tuple[int, ProcessFuture]:
        """Schedule a function for execution and return (job_id, future)."""
        with self._lock:
            if self._pool is None or not self._pool.active:
                context = multiprocessing.get_context(
                    "spawn"
                )  # ensure consistent behavior on Windows (where this is default) and Unix (where fork is default, but that may bring complications)
                self._pool = pebble.ProcessPool(max_workers=self.num_workers, context=context)

            if self._jobs is None:
                self._jobs = {}

            with self._job_id_provider:
                job_id = self._job_id_provider.get_count()
                self._jobs[job_id] = PoolJob(self._pool.schedule(fn, args=args, kwargs=kwargs), user_data)
                if done_callback:
                    self._jobs[job_id].future._waiters.append(ProcessWaiter(job_id, user_data, done_callback))
                # Finally, register our internal cleanup to run last
                self._jobs[job_id].future._waiters.append(ProcessWaiter(job_id, user_data, self._job_done_callback))
                return job_id, self._jobs[job_id].future

    def _job_done_callback(self, _future: ProcessFuture, job_id: int, _user_data: _UserDataT, _state: State) -> None:
        with self._lock:
            if self._jobs is not None and job_id in self._jobs:
                # clean up the work item since we're done with it
                del self._jobs[job_id]

            if self.auto_cleanup_if_no_work:
                # close pool if no work left
                self._cleanup_if_no_jobs()

    def get_job_state(self, job_id: int) -> State | None:
        """Return the current state of the given job, or None if not found."""
        if not self._jobs:
            return None
        job = self._jobs.get(job_id, None)
        if job is None:
            return None
        return _get_status_from_future(job.future)

    def get_job_user_data(self, job_id: int) -> _UserDataT | None:
        """Return the user data associated with a job, or None if not found."""
        if not self._jobs:
            return None
        job = self._jobs.get(job_id, None)
        if job is None:
            return None
        return job.user_data

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a specific job by ID, return True if cancellation succeeded."""
        if not self._jobs:
            return False
        if (job := self._jobs.get(job_id, None)) is None:
            return False
        return job.future.cancel()

    def cancel_all_jobs(self) -> None:
        """Cancel all pending and running jobs in reverse order."""
        if not self._jobs:
            return
        for job_id in reversed(
            self._jobs
        ):  # reversed so that later pending jobs don't start executing when earlier gets cancelled, only to be canceled directly after
            if not self._jobs[job_id].future.done():
                self._jobs[job_id].future.cancel()


class JobPayload(typing.NamedTuple):
    """Encapsulates a callable with its positional and keyword arguments."""

    fn: typing.Callable[..., None]
    args: tuple
    kwargs: dict


class _EMA:
    """Exponential moving average.

    Smoothing to give progressively lower weights to older values.
    N.B.: copied from tqdm

    smoothing  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Ranges from 0 (yields old value) to 1 (yields new value).
    """

    def __init__(self, smoothing: float = 0.3) -> None:
        self.alpha = smoothing
        self.last = 0
        self.calls = 0

    def __call__(self, x: float | None = None) -> float:
        """Update with a new value and return the smoothed result."""
        beta = 1 - self.alpha
        if x is not None:
            self.last = self.alpha * x + beta * self.last
            self.calls += 1
        return self.last / (1 - beta**self.calls) if self.calls else self.last


def _format_interval(t: float) -> str:
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class JobProgress:
    """Lightweight progress tracker (based on tqdm, much reduced functionality)."""

    def __init__(
        self,
        initial: int = 0,
        total: int = 999999,
        unit: str = "it",
        update_interval: int = 1,
        smoothing: float = 0.3,
        printer: typing.Callable[[str], None] | None = None,
        print_interval: int = 100,
    ) -> None:
        """Initialize progress state and EMA smoothers."""
        self.n = initial
        self.total = total
        self.unit = unit
        self.update_interval = update_interval
        self.smoothing = 0.3

        self._printer = printer
        self.print_interval = print_interval

        self._ema_dn = _EMA(smoothing)
        self._ema_dt = _EMA(smoothing)
        self._time = time.time

        self.percentage = 0.0
        self.progress_str = ""

        self.last_update_n = initial
        self.last_update_t = self._time()
        self.start_t = self.last_update_t

    def set_total(self, total: int) -> None:
        """Set the total number of items to process."""
        self.total = total

    def set_unit(self, unit: str) -> None:
        """Set the unit label for progress display."""
        self.unit = unit

    def set_intervals(self, update_interval: int, print_interval: int) -> None:
        """Set the update and print intervals."""
        self.update_interval = max(1, update_interval)
        self.print_interval = max(1, print_interval)

    def update(self, n: int = 1) -> None:
        """Advance progress by ``n`` items and recompute the progress string."""
        self.n += n
        should_print = self._printer is not None and self.n % self.print_interval == 0
        if (
            not self.progress_str
            or self.n - self.last_update_n >= self.update_interval
            or should_print
            or self.n == self.total
        ):
            cur_t = self._time()
            dt = cur_t - self.last_update_t
            dn = self.n - self.last_update_n
            if self.smoothing and dt and dn:
                self._ema_dn(dn)
                self._ema_dt(dt)
                rate = self._ema_dn() / dts if (dts := self._ema_dt()) else None
            else:
                rate = dn / dt if dt else None

            inv_rate = 1 / rate if rate else None
            rate_fmt = (f"{rate:5.2f}" if rate else "?") + " " + self.unit + "/s"
            rate_inv_fmt = (f"{inv_rate:5.2f}" if inv_rate else "?") + " s/" + self.unit
            rate_str = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_fmt

            elapsed = cur_t - self.start_t
            elapsed_str = _format_interval(elapsed)

            remaining = (self.total - self.n) / rate if rate and self.total else 0
            remaining_str = _format_interval(remaining) if rate else "?"

            self.percentage = (self.n / self.total) * 100
            percentage_str = f"{self.percentage:3.0f}%"

            self.progress_str = f"{self.n}/{self.total} ({percentage_str}) [{elapsed_str}<{remaining_str}, {rate_str}]"

            self.last_update_n = self.n
            self.last_update_t = cur_t
        if should_print:
            self._printer(self.progress_str)

    def get_progress(self) -> tuple[float, str]:
        """Return the current (percentage, progress_string) pair."""
        return (self.percentage, self.progress_str)

    def set_start_time_to_now(self) -> None:
        """Reset the start and last-update times to now."""
        self.last_update_t = self._time()
        self.start_t = self.last_update_t

    def set_finished(self) -> None:
        """Mark progress as complete."""
        self.update(self.total - self.n)


multiprocessing.managers.BaseManager.register("JobProgress", JobProgress)


@dataclasses.dataclass
class JobDescription(typing.Generic[_UserDataT]):
    """Full description of a scheduled job including state, progress, and dependencies."""

    user_data: _UserDataT
    payload: JobPayload
    progress: JobProgress
    done_callback: typing.Callable[[ProcessFuture, _UserDataT, int, State], None]

    exclusive_id: int | None = (
        None  # if set, only one task with a given id can be run at a time, rest are kept in waiting
    )
    priority: int = 999  # jobs with a higher priority are scheduled first, unless they cannot be because they're exclusive (exclusive_id is set) and task of that exclusivity is already running, or because their dependencies are not met yet
    depends_on: set[int] | None = None  # set of job ids that need to be completed before this one can be launched

    _pool_job_id: int | None = None
    _future: ProcessFuture | None = dataclasses.field(init=False, default=None)
    _final_state: State | None = dataclasses.field(init=False, default=None)
    error: str | None = None

    def get_state(self) -> State:
        """Return the current lifecycle state of this job."""
        if self._final_state is not None:
            return self._final_state
        if self._future is not None:
            job_state = _get_status_from_future(self._future)
            if job_state not in {State.Pending, State.Running}:
                # finished, cache result
                self._final_state = job_state
                # can also dump the future
                self._future = None
            return job_state
        return State.Pending

    def is_scheduled(self) -> bool:
        """Return True if this job is currently scheduled to the pool."""
        return self._pool_job_id is not None and self.get_state() in {State.Pending, State.Running}

    def is_finished(self) -> bool:
        """Return True if this job has reached a terminal state."""
        return self._final_state is not None


class JobScheduler(typing.Generic[_UserDataT]):
    """Priority-based job scheduler with exclusivity constraints and dependency tracking."""

    def __init__(
        self, pool: ProcessPool, job_is_valid_checker: typing.Callable[[_UserDataT], bool] | None = None
    ) -> None:
        """Initialize the scheduler with a process pool and optional validity checker."""
        self.jobs: dict[int, JobDescription[_UserDataT]] = {}
        self._job_id_provider: CounterContext = CounterContext()
        self._pending_jobs: list[int] = []  # jobs not scheduled or finished

        self._job_is_valid_checker = job_is_valid_checker
        self._pool = pool

        self._manager = multiprocessing.managers.BaseManager(ctx=multiprocessing.get_context("spawn"))
        self._manager.start()

    def add_job(
        self,
        user_data: _UserDataT,
        payload: JobPayload,
        done_callback: typing.Callable[[ProcessFuture, _UserDataT, int, State], None],
        progress_indicator: JobProgress | None = None,
        exclusive_id: int | None = None,
        priority: int | None = None,
        depends_on: set[int] | None = None,
    ) -> int:
        """Add a job to the scheduler and return its ID."""
        with self._job_id_provider:
            job_id = self._job_id_provider.get_count()
        self.jobs[job_id] = JobDescription(
            user_data, payload, progress_indicator, done_callback, exclusive_id, priority, depends_on
        )
        self._pending_jobs.append(job_id)
        return job_id

    def get_progress_indicator(self, **kwargs: typing.Any) -> JobProgress:
        """Create a managed JobProgress instance for cross-process sharing."""
        return self._manager.JobProgress(**kwargs)

    def cancel_job(self, job_id: int) -> None:
        """Cancel a specific job, invoking its callback if not yet pooled."""
        if job_id not in self.jobs:
            return

        if self.jobs[job_id]._pool_job_id is not None:
            self._pool.cancel_job(self.jobs[job_id]._pool_job_id)
        else:
            if job_id in self._pending_jobs:
                self._pending_jobs.remove(job_id)
            self.jobs[job_id]._final_state = State.Canceled
            if self.jobs[job_id].done_callback:
                self.jobs[job_id].done_callback(None, None, self.jobs[job_id].user_data, State.Canceled)
        # TODO: also cancel all jobs that depend on this job

    def cancel_all_jobs(self) -> None:
        """Cancel all jobs and shut down the pool."""
        # cancel any jobs that may still be running
        for job_id in self.jobs:
            if self.jobs[job_id]._final_state not in {State.Completed, State.Canceled, State.Failed}:
                self.cancel_job(job_id)
        # make double sure they're cancelled
        self._pool.cancel_all_jobs()
        # ensure pool is no longer running
        self._pool.cleanup()

    def clear(self) -> None:
        """Cancel all jobs and reset the scheduler to its initial state."""
        # cancel any jobs that may still be running
        self.cancel_all_jobs()
        # clean up
        self._pending_jobs.clear()
        self.jobs.clear()
        # reset job counter
        self._job_id_provider = CounterContext()

    def update(self) -> None:
        """Check job validity, update states, and schedule pending jobs to the pool."""
        # first count how many are scheduled to the pool and whether all tasks are still valid
        num_scheduled_to_pool = 0
        exclusive_ids: set[int] = set()
        for job_id in self.jobs:
            job = self.jobs[job_id]
            # check job still valid or should be canceled
            if self._job_is_valid_checker is not None and not self._job_is_valid_checker(job.user_data):
                self.cancel_job(job_id)
            # remove finished job from pending jobs if its still there
            job.get_state()  # update state
            if job.is_finished() and job_id in self._pending_jobs:
                self._pending_jobs.remove(job_id)
            # check how many scheduled jobs we have
            if job.is_scheduled():
                num_scheduled_to_pool += 1
                if job.exclusive_id is not None:
                    exclusive_ids.add(job.exclusive_id)

        # if we have less than max number of tasks scheduled to the pool, see if anything new to schedule to the pool
        while num_scheduled_to_pool < self._pool.num_workers:
            # find suitable next task to schedule
            # order tasks by priority, filtering out those who have a colliding exclusive_id
            job_ids = [
                i
                for i in sorted(
                    self._pending_jobs,
                    key=lambda ii: 999 if self.jobs[ii].priority is None else self.jobs[ii].priority,
                )
                if self.jobs[i].exclusive_id not in exclusive_ids
            ]
            if not job_ids:
                break
            job_id = job_ids[0]
            to_schedule = self.jobs[job_id]
            extra_kwargs = {}
            if to_schedule.progress is not None:
                extra_kwargs["progress_indicator"] = to_schedule.progress

            to_schedule._pool_job_id, to_schedule._future = self._pool.run(
                to_schedule.payload.fn,
                to_schedule.user_data,
                to_schedule.done_callback,
                *to_schedule.payload.args,
                **extra_kwargs,
                **to_schedule.payload.kwargs,
            )
            self._pending_jobs.remove(job_id)
            if to_schedule.exclusive_id is not None:
                exclusive_ids.add(to_schedule.exclusive_id)
            num_scheduled_to_pool += 1


def _get_status_from_future(fut: ProcessFuture) -> State:
    if fut.running():
        return State.Running
    if fut.done():
        if fut.cancelled():
            return State.Canceled
        if fut.exception() is not None:
            return State.Failed
        return State.Completed
    return State.Pending
