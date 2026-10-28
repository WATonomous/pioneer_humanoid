"""MuJoCo + on-screen keyboard/slider control for the updated Arm V2."""
import json
import math
import queue
import time
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import mujoco
from mujoco import viewer
from articulated_arm import ROOT, build, set_controls, LABELS, LIMITS


def main():
    model, data = build()
    commands = queue.Queue()
    targets = np.zeros(14)
    current = targets.copy()
    selected = 0
    paused = False
    closing = False
    updating = False
    panel = tk.Tk()
    panel.title('Arm V2 - Keyboard Joint Controls')
    width = min(850, panel.winfo_screenwidth()-60)
    height = min(860, panel.winfo_screenheight()-100)
    panel.geometry(f'{width}x{height}+35+35')
    panel.configure(background='#eef2f6')
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background='#eef2f6')
    style.configure('TLabel', background='#eef2f6', foreground='#233b50')
    ttk.Label(panel, text='ARM V2 | ARTICULATED CONTROL', font=('Segoe UI', 17, 'bold')).pack(pady=(14, 4))
    status = tk.StringVar()
    ttk.Label(panel, textvariable=status, font=('Segoe UI', 12, 'bold')).pack(pady=4)
    ttk.Label(panel, text='L / R: select arm    1-7: select joint    Tab: next joint', font=('Segoe UI', 10)).pack()
    ttk.Label(panel, text='Left / Right: -/+ 1 degree    Down / Up: -/+ 5 degrees', font=('Segoe UI', 10)).pack()
    ttk.Label(panel, text='Home: zero selected    Backspace: zero all    Space: pause motion', font=('Segoe UI', 10)).pack()
    area = ttk.Frame(panel)
    area.pack(fill='both', expand=True, padx=14, pady=10)
    variables = []
    labels = []

    def select(index):
        nonlocal selected
        selected = index
        refresh()

    def slider_changed(index, value):
        nonlocal selected
        if updating:
            return
        selected = index
        targets[index] = math.radians(float(value))
        refresh()
    for s, side in enumerate(['RIGHT ARM', 'LEFT ARM']):
        frame = ttk.LabelFrame(area, text=side, padding=10)
        frame.grid(row=0, column=s, sticky='nsew', padx=6)
        area.columnconfigure(s, weight=1)
        for j, title in enumerate(LABELS):
            index = 7*s+j
            label = tk.Label(frame, text=f'{j+1}. {title}', font=('Segoe UI', 10, 'bold'), anchor='w', background='#eef2f6')
            label.pack(fill='x', pady=(6, 0))
            label.bind('<Button-1>', lambda e, i=index: select(i))
            labels.append(label)
            var = tk.DoubleVar(value=0.)
            variables.append(var)
            scale = tk.Scale(frame, from_=LIMITS[j][0], to=LIMITS[j][1], resolution=.5, orient='horizontal',
                             variable=var, command=lambda value, i=index: slider_changed(i, value),
                             length=330, background='#eef2f6', highlightthickness=0, showvalue=True,
                             troughcolor='#d4dce5', activebackground='#537fa6')
            scale.pack(fill='x')
    area.rowconfigure(0, weight=1)
    buttons = ttk.Frame(panel)
    buttons.pack(pady=4)
    ttk.Button(buttons, text='Zero all / CAD pose', command=lambda: command('reset')).pack(side='left', padx=7)
    ttk.Button(buttons, text='Pause / resume', command=lambda: command('pause')).pack(side='left', padx=7)
    ttk.Button(buttons, text='Save pose', command=lambda: command('save')).pack(side='left', padx=7)
    ttk.Label(panel, text='CAD-derived pivots and coupled gripper links. Configurable preview limits.', font=('Segoe UI', 9)).pack(pady=(5, 0))
    ttk.Label(panel, text='Kinematic inspection: torque, mass and contact dynamics are not calibrated.', font=('Segoe UI', 9)).pack(pady=(0, 10))

    def refresh():
        nonlocal updating
        updating = True
        for i, var in enumerate(variables):
            if abs(var.get()-math.degrees(targets[i])) > .01:
                var.set(math.degrees(targets[i]))
            labels[i].configure(background='#bdd7ec' if i == selected else '#eef2f6')
        side = 'RIGHT' if selected < 7 else 'LEFT'
        status.set(f'{side} - {LABELS[selected%7]} | target {math.degrees(targets[selected]):+.1f} deg'+(' | PAUSED' if paused else ''))
        updating = False

    def command(name):
        nonlocal selected, paused
        if name == 'reset':
            targets[:] = 0.
        elif name == 'pause':
            paused = not paused
        elif name == 'save':
            (ROOT/'simulation'/'saved_joint_pose.json').write_text(json.dumps({
                f'{side}_{LABELS[j]}': float(targets[s*7+j]) for s, side in enumerate(['right', 'left']) for j in range(7)}, indent=2))
        refresh()

    def key_action(key):
        nonlocal selected
        if key in ['l', 'L']:
            selected = 7+selected % 7
        elif key in ['r', 'R']:
            selected = selected % 7
        elif key in ['1', '2', '3', '4', '5', '6', '7']:
            selected = 7*(selected//7)+int(key)-1
        elif key in ['Tab']:
            selected = (selected+1) % 14
        elif key == 'ISO_Left_Tab':
            selected = (selected-1) % 14
        elif key in ['Left', 'Right', 'Up', 'Down']:
            amount = {'Left': -1, 'Right': 1, 'Up': 5, 'Down': -5}[key]
            lo, hi = LIMITS[selected % 7]
            targets[selected] = math.radians(np.clip(math.degrees(targets[selected])+amount, lo, hi))
        elif key == 'Home':
            targets[selected] = 0.
        elif key == 'BackSpace':
            command('reset')
        elif key == 'space':
            command('pause')
        refresh()

    def tk_key(event):
        key_action(event.keysym)
        return 'break'
    panel.bind_all('<KeyPress>', tk_key)
    code_map = {258: 'Tab', 263: 'Left', 262: 'Right', 265: 'Up', 264: 'Down', 268: 'Home', 259: 'BackSpace', 32: 'space'}

    def viewer_key(code):
        if code in code_map:
            commands.put(code_map[code])
        elif 32 <= code < 127:
            commands.put(chr(code))
    window = viewer.launch_passive(model, data, key_callback=viewer_key)
    with window.lock():
        window.cam.lookat[:] = [.0216, 0., -.40]
        window.cam.distance = 2.5
        window.cam.azimuth = 25
        window.cam.elevation = -12
    refresh()

    def close():
        nonlocal closing
        if closing:
            return
        closing = True
        command('save')
        window.close()
        panel.destroy()
    panel.protocol('WM_DELETE_WINDOW', close)
    last = time.monotonic()

    def tick():
        nonlocal last
        if closing:
            return
        if not window.is_running():
            close()
            return
        while not commands.empty():
            key_action(commands.get_nowait())
        now = time.monotonic()
        dt = min(.05, now-last)
        last = now
        if not paused:
            current[:] += np.clip(targets-current, -math.radians(45)*dt, math.radians(45)*dt)
        with window.lock():
            set_controls(model, data, current)
            data.time += dt
        window.sync()
        panel.after(16, tick)
    panel.after(16, tick)
    print('Arm V2 keyboard UI running: 14 independent controls, 22 movable joints', flush=True)
    panel.mainloop()


if __name__ == '__main__':
    main()
