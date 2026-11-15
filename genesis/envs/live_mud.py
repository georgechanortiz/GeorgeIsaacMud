"""Mud whose material parameters can be changed while the simulation runs.

Genesis compiles each MPM material's parameters into its kernels as constants, so changing E or nu
normally means rebuilding the scene. LiveLiquid swaps the viscous liquid's stress function for one
that reads mu and lambda from a small field, which can be written between steps without
recompiling. Density is per-particle mass, and the feet's coupling friction is a rigid-solver field,
so both can be changed in place too. Anything that changes the particle count or the compiled
solver (material type, depth, substeps, grid density) still needs a rebuild.
"""

import numpy as np
import quadrants as qd

import genesis as gs


def lame(E, nu):
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


class LiveLiquid:
    """A viscous MPM liquid with live E, nu and rho. Create before the scene is built."""

    def __init__(self, E, nu, rho):
        self.E, self.nu, self.rho = float(E), float(nu), float(rho)
        # always viscous so the solver keeps the SVD the viscous term needs, even if mu is set to 0
        self.material = gs.materials.MPM.Liquid(E=self.E, nu=self.nu, rho=self.rho, viscous=True)
        self.params = qd.field(gs.qd_float, shape=2)  # mu, lambda
        params = self.params

        @qd.func
        def update_stress(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
            return 2 * params[0] * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + qd.Matrix.identity(
                gs.qd_float, 3
            ) * params[1] * J * (J - 1)

        self.material.update_stress = update_stress
        self._write_lame()

    def _write_lame(self):
        mu, lam = lame(self.E, self.nu)
        self.params[0] = mu
        self.params[1] = lam

    def set(self, scene, E=None, nu=None, rho=None):
        """Apply new values between steps."""
        if E is not None or nu is not None:
            self.E = float(E) if E is not None else self.E
            self.nu = float(nu) if nu is not None else self.nu
            self._write_lame()
        if rho is not None and float(rho) != self.rho:
            # the mud is the only MPM material, so every particle's mass scales with the density
            info = scene.sim.mpm_solver.particles_info
            info.mass.from_numpy(info.mass.to_numpy() * (float(rho) / self.rho))
            self.rho = float(rho)


def set_coupling_friction(robot, value):
    """Friction between the robot's coupled geoms and the mud, applied in place."""
    solver = robot.solver
    geoms = solver.dyn_info.geoms
    friction = geoms.coup_friction.to_numpy()
    idx = np.array([g.idx for g in robot.geoms])
    friction[idx] = value
    geoms.coup_friction.from_numpy(friction)


def cfl(E, nu, rho, substep_dt, dx):
    """Pressure-wave CFL number of the liquid; above ~0.5 the explicit MPM step goes unstable."""
    bulk = E / (3.0 * (1.0 - 2.0 * nu))
    return np.sqrt(bulk / rho) * substep_dt / dx
