"""Async-to-thread bridge: run coroutines on a dedicated asyncio event loop."""

import asyncio
import concurrent
import threading
import time
import typing

loop: asyncio.AbstractEventLoop | None = None
thread: threading.Thread | None = None
done_callback: typing.Callable | None = None


def setup(enable_asyncio_debug: bool = False) -> None:
    """Start a background event loop thread (idempotent).

    Args:
        enable_asyncio_debug: If ``True``, enable asyncio debug mode.

    """
    global loop, thread  # noqa: PLW0603
    if loop and thread:
        # already set up, nothing to do
        return

    loop = asyncio.new_event_loop()
    loop.set_debug(enable_asyncio_debug)

    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    thread.start()


def cleanup() -> None:
    """Stop the background event loop and join its thread."""
    global loop, thread  # noqa: PLW0603
    if loop:
        loop.call_soon_threadsafe(loop.stop)
        loop = None
    if thread:
        thread.join()
        thread = None


def run(
    coroutine: typing.Coroutine, override_done_callback: typing.Callable | None = None
) -> concurrent.futures.Future:
    """Submit a coroutine to the background loop and return a Future.

    Args:
        coroutine: The coroutine to schedule.
        override_done_callback: Optional callback; if ``None``, uses the
            module-level ``done_callback``.

    Returns:
        A ``Future`` representing the coroutine's eventual result.

    """
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    if override_done_callback:
        future.add_done_callback(override_done_callback)
    elif done_callback:
        future.add_done_callback(done_callback)
    return future


def wait(coroutine: typing.Coroutine) -> typing.Any:
    """Submit a coroutine and block until it completes, returning its result.

    Args:
        coroutine: The coroutine to schedule.

    Returns:
        The coroutine's return value.

    Raises:
        Exception: Re-raises any exception from the coroutine.

    """
    future = run(coroutine)
    while future.running():
        time.sleep(0.1)
    if exception := future.exception():
        raise exception
    return future.result()


# Example usage
if __name__ == "__main__":
    import async_thread  # This script is designed as a module you import

    async_thread.setup()

    import random

    async def wait_and_say_hello(num: int) -> None:
        """Print a greeting after a random delay."""
        await asyncio.sleep(random.random())
        print(f"Hello {num}!")

    for i in range(10):
        async_thread.run(wait_and_say_hello(i))

    # You can also wait for the task to complete:
    for i in range(10):
        async_thread.wait(wait_and_say_hello(i))
