"""A MuJoCo scene in a plain GLFW window that also reads the keyboard, for the MuJoCo teleops. Sim only.

Not mujoco.viewer: it maps every letter to a display toggle that can't be turned off (W wireframe, S shadows,
D hides the table, ...), so teleop and recording keys change the picture. Mouse: left drag orbits, right drag
pans, wheel zooms.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np


class SceneWindow:
    """Held letters + Shift, and press events for the `one_shot` keys (and "esc"). sync() once per control
    step takes in the events. A thread draws, as in mujoco.viewer (a frame with shadows can take longer than a
    control step): hold `lock` while changing `data`."""

    def __init__(self, model, data, title: str, camera: dict | None = None, one_shot: str = ""):
        import glfw
        import mujoco

        self.glfw, self.mj, self.m, self.d = glfw, mujoco, model, data
        self.one_shot = set(one_shot)
        self.held: set[str] = set()
        self.shift = False
        self.events: deque[str] = deque()
        self.overlay = ""
        self.decorate = None             # optional fn(scene): add marker geoms to each frame (called with `lock` held)
        self.lock = threading.Lock()
        if not glfw.init():
            raise RuntimeError("could not initialise GLFW (no display?)")
        glfw.window_hint(glfw.VISIBLE, 1)        # mujoco.Renderer's offscreen contexts leave this hint at 0
        glfw.window_hint(glfw.SAMPLES, 4)
        self.win = glfw.create_window(1280, 800, title, None, None)
        glfw.default_window_hints()
        if not self.win:
            raise RuntimeError("could not open the scene window")
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        for k, v in (camera or {}).items():
            setattr(self.cam, k, v)
        self.scn = mujoco.MjvScene(model, maxgeom=10000)
        self._mouse = glfw.get_cursor_pos(self.win)
        self._size = glfw.get_framebuffer_size(self.win)
        glfw.set_key_callback(self.win, self._key)
        glfw.set_window_focus_callback(self.win, self._focus)
        glfw.set_cursor_pos_callback(self.win, self._cursor)
        glfw.set_scroll_callback(self.win, self._scroll)
        glfw.set_framebuffer_size_callback(self.win, self._resized)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._draw_loop, daemon=True)
        self._thread.start()

    def _draw_loop(self):
        glfw, mj = self.glfw, self.mj
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        ctx = mj.MjrContext(self.m, mj.mjtFontScale.mjFONTSCALE_150)
        while not self._stop.is_set():
            width, height = self._size
            if not (width and height):             # 0 x 0 while minimised
                time.sleep(0.05)
                continue
            with self.lock:
                mj.mjv_updateScene(self.m, self.d, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL, self.scn)
                if self.decorate is not None:
                    self.decorate(self.scn)
            viewport = mj.MjrRect(0, 0, width, height)
            mj.mjr_render(viewport, self.scn, ctx)
            if self.overlay:
                mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT, viewport,
                               self.overlay, "", ctx)
            glfw.swap_buffers(self.win)
        ctx.free()
        glfw.make_context_current(None)

    def _resized(self, win, width, height):
        self._size = (width, height)

    def _key(self, win, key, scancode, action, mods):
        glfw = self.glfw
        if key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT):
            self.shift = action != glfw.RELEASE
        elif key == glfw.KEY_ESCAPE:
            if action == glfw.PRESS:
                self.events.append("esc")
        elif glfw.KEY_A <= key <= glfw.KEY_Z:
            c = chr(key).lower()
            if action == glfw.PRESS:               # not REPEAT: a held key acts once
                self.held.add(c)
                if c in self.one_shot:
                    self.events.append(c)
            elif action == glfw.RELEASE:
                self.held.discard(c)

    def _focus(self, win, focused):
        if not focused:                            # the releases go to another window: don't keep moving
            self.held.clear()
            self.shift = False

    def _cursor(self, win, x, y):
        # By hand, not mjv_moveCamera: its arguments differ between MuJoCo versions.
        glfw, cam = self.glfw, self.cam
        height = max(glfw.get_window_size(win)[1], 1)
        dx, dy = (x - self._mouse[0]) / height, (y - self._mouse[1]) / height
        self._mouse = (x, y)
        with self.lock:
            if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                cam.azimuth -= 180.0 * dx
                cam.elevation = float(np.clip(cam.elevation - 180.0 * dy, -89.0, 89.0))
            elif glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
                forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
                right = np.array([math.sin(az), -math.cos(az), 0.0])
                cam.lookat[:] += cam.distance * (dy * np.cross(right, forward) - dx * right)

    def _scroll(self, win, dx, dy):
        with self.lock:
            self.cam.distance = max(0.05, self.cam.distance * 0.9 ** dy)

    def is_running(self) -> bool:
        return self._thread.is_alive() and not self.glfw.window_should_close(self.win)

    def sync(self, overlay: str = "") -> None:
        """Take in the key and mouse events; `overlay`: text for the top left corner."""
        self.overlay = overlay
        self.glfw.poll_events()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self.glfw.destroy_window(self.win)
