"""File system browsing provider with callback-driven directory listings and actions.

Wraps :mod:`file_actions` functions with a callback pattern suitable for GUI
applications where directory listings and file operations need to notify the
UI upon completion.
"""

import concurrent
import pathlib
from collections.abc import Callable
from typing import Any

from . import async_thread, file_actions, platform

# Callback type for directory listing results: (path, result_or_error)
_ListingCallback = Callable[[str | pathlib.Path, list[file_actions.DirEntry] | Exception], None]
# Callback type for file action results: (path, action_name, result_or_error)
_ActionCallback = Callable[[pathlib.Path, str, pathlib.Path | Exception | None], None]


class FileActionProvider:
    """Provides directory listing, creation, and renaming with callback support.

    Callbacks are invoked when operations complete (successfully or with errors).
    Multiple callbacks can be registered for each operation type.

    Attributes:
        local_name: Display name for the filesystem root ("This PC" on Windows,
            "Root" elsewhere).

    """

    def __init__(
        self,
        listing_callback: _ListingCallback | None = None,
        action_callback: _ActionCallback | None = None,
    ) -> None:
        """Initialize with optional listing and action callbacks.

        Args:
            listing_callback: Called when a directory listing completes. Receives
                the requested path and either a list of entries or an exception.
            action_callback: Called when a file action (mkdir, rename) completes.
                Receives the target path, the action name, and either the result
                or an exception.

        """
        self.waiters: set[concurrent.futures.Future] = set()

        self.listing_callbacks: list[_ListingCallback] = []
        if listing_callback:
            self.listing_callbacks.append(listing_callback)
        self.action_callbacks: list[_ActionCallback] = []
        if action_callback:
            self.action_callbacks.append(action_callback)

    def __del__(self) -> None:
        """Cancel any pending futures on cleanup."""
        for w in self.waiters:
            if not w.done():
                w.cancel()

    local_name = "This PC" if platform.os == platform.Os.Windows else "Root"

    def get_listing(self, path: str | pathlib.Path) -> None:
        r"""Retrieve a directory listing and deliver it via listing callbacks.

        On Windows, the special path ``"root"`` returns drives and "This PC"
        folders (Desktop, Documents, Downloads). Bare network computer paths
        (e.g. ``\\SERVER``) return visible shares. All other paths use a
        standard directory listing.

        On other platforms, ``"root"`` maps to ``/``.

        Args:
            path: Directory path to list, or ``"root"`` for the filesystem root.

        """
        if platform.os == platform.Os.Windows:
            if path == "root":
                try:
                    result = file_actions.get_drives()
                    result.extend(file_actions.get_thispc_listing())
                except Exception as exc:
                    result = exc
                self._listing_done(result, "root")
            else:
                net_comp = file_actions.get_net_computer(path)
                try:
                    if net_comp:
                        # bare network computer name (e.g. \\SERVER), get its shares
                        result = file_actions.get_visible_shares(net_comp, "Guest", "")
                    else:
                        # normal directory or share on a network computer
                        result = file_actions.get_dir_list_sync(path)
                except Exception as exc:
                    result = exc
                self._listing_done(result, path)
        else:
            try:
                l_path = path
                if path == "root":
                    l_path = "/"
                result = file_actions.get_dir_list_sync(l_path)
            except Exception as exc:
                result = exc
            self._listing_done(result, path)

    def _listing_done(
        self,
        result_or_fut: concurrent.futures.Future | list[file_actions.DirEntry] | Exception,
        path: str | pathlib.Path,
    ) -> None:
        """Unwrap a listing result and notify all listing callbacks.

        Args:
            result_or_fut: Either a completed Future to unwrap or a direct
                result (entry list / exception) from a synchronous listing.
            path: The path that was listed (passed through to callbacks).

        """
        result = self._get_result_from_future(result_or_fut)
        if result == "cancelled":
            return

        if not self.listing_callbacks:
            return
        if result is None:
            return

        for c in self.listing_callbacks:
            c(path, result)

    def make_dir(self, path: pathlib.Path) -> concurrent.futures.Future:
        """Create a directory asynchronously via :func:`file_actions.make_dir`.

        Args:
            path: Directory path to create.

        Returns:
            Future that completes when the directory is created. Action
            callbacks are notified on completion.

        """
        action = "make_dir"
        fut = async_thread.run(file_actions.make_dir(path), lambda f: self._action_done(f, path, action))
        self.waiters.add(fut)
        return fut

    def rename_path(self, old_path: pathlib.Path, new_path: pathlib.Path) -> concurrent.futures.Future:
        """Rename a path asynchronously via :func:`file_actions.rename_path`.

        Args:
            old_path: Current path.
            new_path: Desired new path.

        Returns:
            Future that completes when the rename is done. Action callbacks
            are notified on completion.

        """
        action = "rename_path"
        fut = async_thread.run(
            file_actions.rename_path(old_path, new_path), lambda f: self._action_done(f, old_path, action)
        )
        self.waiters.add(fut)
        return fut

    def _action_done(self, fut: concurrent.futures.Future, path: pathlib.Path, action: str) -> None:
        """Unwrap an action result and notify all action callbacks.

        Args:
            fut: Completed Future from :func:`async_thread.run`.
            path: The target path of the action (passed through to callbacks).
            action: Action identifier (e.g. ``"make_dir"``, ``"rename_path"``).

        """
        result = self._get_result_from_future(fut)
        if result == "cancelled":
            return

        if not self.action_callbacks:
            return

        for c in self.action_callbacks:
            c(path, action, result)

    def _get_result_from_future(self, fut: concurrent.futures.Future | list[file_actions.DirEntry] | Exception) -> Any:
        """Extract the result from a Future or pass through a direct value.

        If *fut* is a :class:`~concurrent.futures.Future`, removes it from
        :attr:`waiters` and returns the result, the exception, or the string
        ``"cancelled"``. Otherwise returns *fut* unchanged.

        Args:
            fut: A Future to unwrap, or a direct result value.

        Returns:
            The unwrapped result, an exception, or ``"cancelled"``.

        """
        if isinstance(fut, concurrent.futures.Future):
            self.waiters.discard(fut)
            try:
                return fut.result()
            except concurrent.futures.CancelledError:
                return "cancelled"
            except Exception as exc:
                return exc
        else:
            return fut
