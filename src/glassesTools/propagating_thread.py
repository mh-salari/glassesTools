"""Thread subclass that propagates exceptions from the worker back to the caller."""

# from https://stackoverflow.com/a/31614591/3103767
# added cleanup fun that may be needed to cancel work occurring in main thread so that join gets called
import typing
from collections.abc import Callable
from threading import Thread


class PropagatingThread(Thread):
    """Thread that re-raises worker exceptions on join()."""

    def __init__(self, cleanup_fun: Callable | None = None, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Initialize with an optional cleanup callback invoked on worker failure.

        Args:
            cleanup_fun: Called if the target function raises an exception.
            *args: Passed through to ``threading.Thread``.
            **kwargs: Passed through to ``threading.Thread``.

        """
        super().__init__(*args, **kwargs)
        self.cleanup_fun = cleanup_fun

    def run(self) -> None:
        """Execute the target function, capturing any exception."""
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e
            if self.cleanup_fun:
                self.cleanup_fun()

    def join(self, timeout: float | None = None) -> typing.Any:
        """Wait for the thread to finish and re-raise any captured exception.

        Args:
            timeout: Maximum seconds to wait (``None`` = wait forever).

        Returns:
            The target function's return value.

        Raises:
            BaseException: Re-raises any exception from the worker thread.

        """
        super().join(timeout)
        if self.exc:
            raise self.exc
        return self.ret
