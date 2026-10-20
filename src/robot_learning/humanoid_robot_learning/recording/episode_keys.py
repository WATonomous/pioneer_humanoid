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
    Register keys:
      S — start / resume logging frames for the current episode
      N — finish episode (save on next loop tick)
      D — discard buffered frames and re-record current episode
      ESC — stop session and finalize dataset
    """

    def __init__(self, flags: EpisodeFlags) -> None:
        self.flags = flags
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
                    self.flags.abort = True
                    logger.info("[ESC] Abort recording.")
                    return False
                char = getattr(key, "char", None)
                if char is None:
                    return
                c = char.lower()
                if c == "s":
                    self.flags.start = True
                    logger.info("[S] Recording frames for this episode.")
                elif c == "n":
                    self.flags.success = True
                    logger.info("[N] Mark episode complete.")
                elif c == "d":
                    self.flags.remove = True
                    logger.info("[D] Discard episode buffer.")
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
