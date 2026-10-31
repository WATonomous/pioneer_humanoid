"""Keyboard episode controls (lehome dataset_record.py key bindings)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EpisodeFlags:
    """Shared flags between keyboard thread and record loop."""

    start: bool = False
    success: bool = False
    remove: bool = False
    abort: bool = False


class EpisodeKeyboard:
    """
    Register keys (defaults; a viewer that already uses them can pass others):
      S — start / resume logging frames for the current episode
      N — finish episode (save on next loop tick)
      D — discard buffered frames and re-record current episode
      ESC — stop session and finalize dataset (esc_aborts=False ignores it)
    """

    def __init__(self, flags: EpisodeFlags, start: str = "s", save: str = "n", discard: str = "d",
                 esc_aborts: bool = True) -> None:
        self.flags = flags
        self._keys = {start: "start", save: "success", discard: "remove"}
        self._esc_aborts = esc_aborts
        self._listener = None
        self._pynput = None

    def start(self) -> bool:
        try:
            from pynput import keyboard as kb
        except ImportError:
            logger.warning(
                "pynput not installed; keyboard controls disabled. "
                "Use timed mode (--episode_time_s) or pip install pynput."
            )
            return False

        self._pynput = kb

        def on_press(key):
            try:
                if key == kb.Key.esc:
                    if not self._esc_aborts:
                        return
                    self.flags.abort = True
                    logger.info("[ESC] Abort recording.")
                    return False
                char = getattr(key, "char", None)
                flag = self._keys.get(char.lower()) if char else None
                if flag is not None:
                    setattr(self.flags, flag, True)
                    logger.info("[%s] %s", char.upper(), flag)
            except Exception:
                logger.exception("Keyboard callback error")

        self._listener = kb.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
