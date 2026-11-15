"""Control panel for scripts/mud_tuner_a2.py. Started by it; not meant to be run directly.

Talks to the simulator over pipes, one JSON object per line: commands go out on stdout, status
comes in on stdin. Live sliders apply when released; the rebuild fields restart the simulator.
"""

import json
import math
import sys
import threading
import tkinter as tk
from tkinter import ttk

CONTROL_DT = 0.02  # s, the policy step


def send(**msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


class Panel:
    def __init__(self, root, init):
        self.root = root
        self.init = init
        root.title("Mud tuner")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.vars = {}

        live = ttk.LabelFrame(root, text="Live mud (applied on release)")
        live.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)
        self._slider(live, 0, "log10 E (Pa)", "logE", 3.5, 6.7, math.log10(init["E"]), 0.05)
        self._slider(live, 1, "Poisson ratio nu", "nu", 0.05, 0.45, init["nu"], 0.01)
        self._slider(live, 2, "Density (kg/m3)", "rho", 1000, 2200, init["rho"], 10)
        self._slider(live, 3, "Foot-mud friction", "friction", 0.0, 1.5, init["friction"], 0.05)
        self._slider(live, 4, "Commanded vx (m/s)", "vx", 0.0, 1.2, init["vx"], 0.05)
        self.derived = ttk.Label(live, text="", justify="left")
        self.derived.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        build = ttk.LabelFrame(root, text="Needs a rebuild (restarts the simulator)")
        build.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        self.depth = self._entry(build, 0, "Mud depth (m)", init["depth"])
        self.substeps = self._entry(build, 1, "Substeps (multiple of 4)", init["substeps"])
        self.grid_density = self._entry(build, 2, "Grid density (cells/m)", init["grid_density"])
        ttk.Button(build, text="Rebuild", command=self.rebuild).grid(row=3, column=0, columnspan=2, pady=4)

        ctrl = ttk.Frame(root)
        ctrl.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        ttk.Button(ctrl, text="Restart", command=lambda: send(cmd="restart")).grid(row=0, column=0, padx=2)
        self.pause_btn = ttk.Button(ctrl, text="Pause", command=self.toggle_pause)
        self.pause_btn.grid(row=0, column=1, padx=2)
        ttk.Button(ctrl, text="Calibrated mud", command=self.calibrated).grid(row=0, column=2, padx=2)
        self.auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Restart at pit end", variable=self.auto,
                        command=lambda: send(cmd="auto_restart", value=self.auto.get())).grid(row=1, column=0, columnspan=2, sticky="w")
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Viewer follows robot", variable=self.follow,
                        command=lambda: send(cmd="follow", value=self.follow.get())).grid(row=1, column=2, sticky="w")
        self.paused = False

        status = ttk.LabelFrame(root, text="Status")
        status.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)
        self.status = ttk.Label(status, text="building scene...", justify="left", font=("TkFixedFont", 10))
        self.status.grid(row=0, column=0, sticky="w")

        self._update_derived()
        threading.Thread(target=self._read_status, daemon=True).start()

    # ---- widgets ----
    def _slider(self, parent, row, label, key, lo, hi, value, step):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        var = tk.DoubleVar(value=value)
        self.vars[key] = var
        shown = ttk.Label(parent, width=10)
        shown.grid(row=row, column=2, sticky="e")

        def on_move(_=None):
            v = round(var.get() / step) * step
            var.set(v)
            shown.config(text=f"{10 ** v:.3g}" if key == "logE" else f"{v:.3g}")
            self._update_derived()

        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, length=260, command=on_move)
        scale.grid(row=row, column=1, padx=4)
        scale.bind("<ButtonRelease-1>", lambda _: self._apply(key))
        on_move()

    def _entry(self, parent, row, label, value):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=str(value))
        ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, sticky="w", padx=4)
        return var

    # ---- actions ----
    def _value(self, key):
        v = self.vars[key].get()
        return 10 ** v if key == "logE" else v

    def _apply(self, key):
        name = "E" if key == "logE" else key
        send(cmd="set", name=name, value=self._value(key))

    def _update_derived(self):
        if not hasattr(self, "derived"):  # sliders are built before the label
            return
        try:
            E, nu, rho = self._value("logE"), self._value("nu"), self._value("rho")
            substeps = int(self.substeps.get()) if hasattr(self, "substeps") else self.init["substeps"]
            dx = 1.0 / (int(self.grid_density.get()) if hasattr(self, "grid_density") else self.init["grid_density"])
        except (ValueError, KeyError):
            return
        dt = CONTROL_DT / substeps
        mu = E / (2 * (1 + nu))
        cfl = math.sqrt(E / (3 * (1 - 2 * nu)) / rho) * dt / dx
        warn = "  UNSTABLE: raise substeps" if cfl > 0.5 else ("  near limit" if cfl > 0.35 else "")
        self.derived.config(
            text=f"effective viscosity mu*dt = {mu * dt:.3g} Pa*s   (calibrated: {self.init['calibrated_visc']:.3g})\n"
                 f"stability (CFL) = {cfl:.2f}{warn}"
        )

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
        send(cmd="pause", value=self.paused)

    def calibrated(self):
        for key, value in (("logE", math.log10(1e5)), ("nu", 0.35), ("rho", 1500.0), ("friction", 0.4)):
            self.vars[key].set(value)
            self._apply(key)
        self._update_derived()

    def rebuild(self):
        try:
            substeps = int(self.substeps.get())
            depth = float(self.depth.get())
            grid_density = int(self.grid_density.get())
        except ValueError:
            self.status.config(text="rebuild: depth must be a number, substeps and grid density integers")
            return
        if substeps % 4:
            self.status.config(text="rebuild: substeps must be a multiple of 4")
            return
        send(cmd="rebuild", depth=depth, substeps=substeps, grid_density=grid_density,
             E=self._value("logE"), nu=self._value("nu"), rho=self._value("rho"),
             friction=self._value("friction"), vx=self._value("vx"))

    def quit(self):
        send(cmd="quit")
        self.root.destroy()

    # ---- status from the simulator ----
    def _read_status(self):
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.root.after(0, self._show_status, msg)
        self.root.after(0, self.root.destroy)  # simulator exited

    def _show_status(self, msg):
        if "text" in msg:
            self.status.config(text=msg["text"])


def main():
    init = json.loads(sys.argv[1])
    root = tk.Tk()
    Panel(root, init)
    root.mainloop()


if __name__ == "__main__":
    main()
