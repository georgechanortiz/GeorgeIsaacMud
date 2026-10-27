"""Building blocks for an MPM mud pit, and the pits the eval scripts use.

Shared by the policy-driven MPM envs (envs/mpm_env.py), the sphere drop bench
(scripts/eval_mpm_1d_intrusion.py) and the mud calibration (scripts/mud_calibration/run_one.py),
so they all build the same mud the same way.

An mpm_cfg is a dict:

    grid_density                  MPM cells per metre
    coup_friction, coup_softness  friction and reach of the foot <-> mud coupling
    wall_height, wall_thickness   containment walls around the patches
    bounds_lower, bounds_upper    corners of the MPM domain
    patches                       list of material boxes: name, material, E, nu, rho,
                                  patch_lower, patch_upper, color, plus per-material keys
                                  (yield_stress, friction_angle, viscous)
"""

import numpy as np

import genesis as gs

# The hardware pit: 12 ft long and 32 in wide inside its walls, with the MPM domain around it.
PIT_X = (1.1, 4.55)
PIT_HALF_WIDTH = 0.38
_BOUNDS_LOWER = (1, -0.4864, -0.1)
_BOUNDS_UPPER = (4.6576, 0.4864, 0.381)


def _pit(grid_density, patches, wall_height, coup_friction=0.4, coup_softness=0.01,
         bounds_lower=_BOUNDS_LOWER, bounds_upper=_BOUNDS_UPPER):
    return {
        "grid_density": grid_density,    # cells per metre. higher = finer particles, costs speed (~cubic)
        "coup_friction": coup_friction,  # higher = more grip between the feet and the material
        "coup_softness": coup_softness,  # metres; reach of the foot<->material coupling, see coupling_material
        "wall_height": wall_height,      # m, from the ground up; taller than the mud + robot stance
        "wall_thickness": 0.03,          # m
        "bounds_lower": bounds_lower,
        "bounds_upper": bounds_upper,
        "patches": patches,
    }


def go2_mud_pit(soft_layer, grid_density=48):
    """The Go2 pit: a firm elastic base 6 cm deep, with `soft_layer` (one patch dict) on top."""
    stiff_base = {
        "name": "stiff_mud",
        "material": "elastic",  # firm base layer to hold the robot's weight
        "E": 1e6,               # Young's modulus (Pa)
        "nu": 0.35,             # Poisson ratio. higher = more volume-preserving; must stay < 0.5 (lam blows up near it)
        "rho": 1800.0,          # kg/m3. Sets particle inertia / elastic wave speed, not force magnitude (that's E)
        "patch_lower": (PIT_X[0], -PIT_HALF_WIDTH, 0.0),
        "patch_upper": (PIT_X[1], PIT_HALF_WIDTH, 0.06),
        "color": (0.30, 0.20, 0.12),
    }
    return _pit(grid_density, [stiff_base, soft_layer], wall_height=0.5)


def soft_layer(material, z_range, **params):
    """A soft top layer spanning the pit at heights z_range = (low, high)."""
    return {
        "name": "soft_mud",
        "material": material,
        **params,
        "patch_lower": (PIT_X[0], -PIT_HALF_WIDTH, z_range[0]),
        "patch_upper": (PIT_X[1], PIT_HALF_WIDTH, z_range[1]),
        "color": (0.78, 0.62, 0.46),
    }


def a2_mud_pit(mud, depth, grid_density=48, coup_friction=0.4, coup_softness=0.01):
    """The A2 pit: one layer of `mud` (material name and parameters) lying directly on the floor.

    As in the hardware runs, the feet sink through it to the firm bottom. The walls are 0.6 m
    high, since the A2 stands taller than the Go2's 0.5 m walls allowed for.
    """
    layer = {
        "name": "mud",
        **mud,
        "patch_lower": (PIT_X[0], -PIT_HALF_WIDTH, 0.0),
        "patch_upper": (PIT_X[1], PIT_HALF_WIDTH, depth),
        "color": (0.55, 0.42, 0.30),
    }
    return _pit(grid_density, [layer], wall_height=0.6, coup_friction=coup_friction, coup_softness=coup_softness)


def drop_pit(layer, side=0.6096, depth=0.1524, grid_density=48, wall_height=0.2):
    """A square pit `side` x `side` (default 2 ft) of `layer` filled `depth` deep (default 6 in), centred on the origin.

    Unlike the hardware pit, the MPM domain hugs the patch: its usable region (bounds inset by
    3 cells) ends right at the patch's sides, so the domain boundary is what holds the material
    in -- the rigid walls are needs_coup=False and only visual. The 1 mm slack keeps float
    rounding from tripping patch_box's bounds check.
    """
    half = side / 2.0
    lateral = half + 3.0 / grid_density + 1e-3
    patch = {
        "name": layer.get("name", "mud"),
        **layer,
        "patch_lower": (-half, -half, 0.0),
        "patch_upper": (half, half, depth),
        "color": layer.get("color", (0.78, 0.62, 0.46)),
    }
    return _pit(
        grid_density, [patch], wall_height=wall_height,
        bounds_lower=(-lateral, -lateral, -0.1), bounds_upper=(lateral, lateral, 0.4),
    )


def surface_height(mpm_cfg):
    """Top of the highest patch: where a robot spawned on the pit stands."""
    return max(p["patch_upper"][2] for p in mpm_cfg["patches"])


def spawn_on_pit(env_cfg, mpm_cfg, offset):
    """Spawn the robot `offset` metres past the pit's leading edge, lifted onto the mud surface."""
    _, by, bz = env_cfg["base_init_pos"]
    x_lo = min(p["patch_lower"][0] for p in mpm_cfg["patches"])
    env_cfg["base_init_pos"] = [x_lo + offset, by, bz + surface_height(mpm_cfg)]


def pit_end_x(mpm_cfg, margin):
    """Base x at which the robot counts as having crossed: `margin` short of the far wall."""
    return max(p["patch_upper"][0] for p in mpm_cfg["patches"]) - margin


# ---- scene construction --------------------------------------------------------------------------


def mpm_options(mpm_cfg):
    return gs.options.MPMOptions(
        lower_bound=mpm_cfg["bounds_lower"],
        upper_bound=mpm_cfg["bounds_upper"],
        grid_density=mpm_cfg["grid_density"],
        # feet are smaller than a grid cell; CPIC is needed for small rigid bodies to couple properly
        enable_CPIC=True,
    )


def coupling_material(mpm_cfg):
    """Rigid material for the floor and robot, coupled to the mud.

    needs_coup is already the default, but coupling is the point here, so it is explicit.
    coup_softness widens the coupling reach past the default 2 mm: the grid spacing (~21 mm at
    grid_density 48) is otherwise too coarse to catch the small feet.
    """
    return gs.materials.Rigid(
        needs_coup=True,
        coup_friction=mpm_cfg["coup_friction"],
        coup_softness=mpm_cfg["coup_softness"],
    )


def mpm_material(patch):
    """The Genesis MPM material a patch dict describes."""
    kind, E, nu, rho = patch["material"], patch["E"], patch["nu"], patch["rho"]
    if kind == "elastic":
        return gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho)
    if kind == "sand":
        return gs.materials.MPM.Sand(E=E, nu=nu, rho=rho, friction_angle=patch["friction_angle"])
    if kind == "liquid":
        # viscous=True gives a shear stress 2*mu*(F - R)F^T, with mu set by E and nu
        return gs.materials.MPM.Liquid(E=E, nu=nu, rho=rho, viscous=patch.get("viscous", True))
    if kind == "snow":
        return gs.materials.MPM.Snow(E=E, nu=nu, rho=rho)
    if kind == "elasto_plastic":
        return gs.materials.MPM.ElastoPlastic(
            E=E, nu=nu, rho=rho, use_von_mises=True, von_mises_yield_stress=patch["yield_stress"]
        )
    raise ValueError(f"patch '{patch.get('name')}': unknown MPM material {kind!r}")


def patch_box(patch, mpm_cfg):
    """Resolve one patch's patch_lower/patch_upper into the (pos, size) gs.morphs.Box wants.

    The MPM solver insets its usable region by 3 grid cells on every axis, so the
    valid spawn region shrinks as grid_density drops. The solver rejects out-of-bounds
    entities but doesn't say which axis overflowed, so check it here first.
    """
    inset = 3.0 / mpm_cfg["grid_density"]
    label = patch.get("name", "patch")
    lower = np.asarray(patch["patch_lower"], dtype=float)
    upper = np.asarray(patch["patch_upper"], dtype=float)
    bounds_lower = np.asarray(mpm_cfg["bounds_lower"], dtype=float) + inset
    bounds_upper = np.asarray(mpm_cfg["bounds_upper"], dtype=float) - inset

    for axis, name in enumerate("xyz"):
        if upper[axis] <= lower[axis]:
            raise ValueError(
                f"patch '{label}': patch_upper must exceed patch_lower on {name}: "
                f"got {lower[axis]:.3f} .. {upper[axis]:.3f}"
            )
        if lower[axis] < bounds_lower[axis] or upper[axis] > bounds_upper[axis]:
            raise ValueError(
                f"patch '{label}' overflows the MPM domain on {name}: patch spans "
                f"[{lower[axis]:.3f}, {upper[axis]:.3f}] but the usable region is "
                f"[{bounds_lower[axis]:.3f}, {bounds_upper[axis]:.3f}] "
                f"(bounds inset by 3/grid_density = {inset:.4f}). "
                f"Widen bounds_{'lower' if lower[axis] < bounds_lower[axis] else 'upper'}, "
                f"raise grid_density, or shrink the patch."
            )

    pos = tuple(float(v) for v in (lower + upper) / 2)
    size = tuple(float(v) for v in (upper - lower))
    return pos, size


def check_no_overlap(patches):
    """Overlapping patches sample particles into the same cells -> instant blow-up.

    Two axis-aligned boxes overlap iff they overlap on all three axes.
    """
    boxes = [
        (np.asarray(p["patch_lower"], float), np.asarray(p["patch_upper"], float), p.get("name", i))
        for i, p in enumerate(patches)
    ]
    for i in range(len(boxes)):
        la, ua, ni = boxes[i]
        for j in range(i + 1, len(boxes)):
            lb, ub, nj = boxes[j]
            if np.all(la < ub) and np.all(lb < ua):
                raise ValueError(
                    f"MPM patches '{ni}' and '{nj}' overlap; give them disjoint "
                    f"patch_lower/patch_upper boxes (a gap on any one axis is enough)."
                )


def add_mud_patches(scene, mpm_cfg):
    """Add one entity per patch, all sharing the scene's single MPM solver. Returns them in config order."""
    patches = mpm_cfg["patches"]
    check_no_overlap(patches)
    entities = []
    for patch in patches:
        pos, size = patch_box(patch, mpm_cfg)
        entities.append(
            scene.add_entity(
                material=mpm_material(patch),
                morph=gs.morphs.Box(pos=pos, size=size),
                surface=gs.surfaces.Default(color=patch.get("color", (0.35, 0.25, 0.15)), vis_mode="particle"),
            )
        )
    return entities


def add_containment_walls(scene, mpm_cfg):
    """Thin fixed rigid walls around the patch perimeter so the robot can't walk off the pit.

    needs_coup=False keeps the walls invisible to the MPM solver -- they only block
    the robot, not the mud. Surface is semi-transparent so the near wall doesn't
    occlude the camera.
    """
    lowers = np.asarray([p["patch_lower"] for p in mpm_cfg["patches"]], dtype=float)
    uppers = np.asarray([p["patch_upper"] for p in mpm_cfg["patches"]], dtype=float)
    x_lo, y_lo = lowers[:, 0].min(), lowers[:, 1].min()
    x_hi, y_hi = uppers[:, 0].max(), uppers[:, 1].max()
    t = float(mpm_cfg["wall_thickness"])
    h = float(mpm_cfg["wall_height"])
    cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
    x_len, y_len = x_hi - x_lo, y_hi - y_lo
    # (pos, size); the +/-y boards run the full length so the four corners close.
    boards = [
        ((x_lo - t / 2, cy, h / 2), (t, y_len, h)),          # near (-x), behind the spawn
        ((x_hi + t / 2, cy, h / 2), (t, y_len, h)),          # far (+x), the edge it marches toward
        ((cx, y_lo - t / 2, h / 2), (x_len + 2 * t, t, h)),  # -y side
        ((cx, y_hi + t / 2, h / 2), (x_len + 2 * t, t, h)),  # +y side
    ]
    wall_material = gs.materials.Rigid(needs_coup=False)
    return [
        scene.add_entity(
            material=wall_material,
            morph=gs.morphs.Box(pos=pos, size=size, fixed=True),
            surface=gs.surfaces.Default(color=(0.55, 0.57, 0.62), opacity=0.2, vis_mode="visual"),
        )
        for pos, size in boards
    ]


def coupling_force(scene, links_idx):
    """Force the MPM material exerts on each of the given rigid links, world frame, shape (num_envs, num_links, 3).

    get_links_net_contact_force() only reports rigid-rigid contacts, which are ~zero for a body
    resting on MPM material -- the material pushes on it through the MPM<->rigid coupler instead,
    landing in dyn_state.links.cfrc_coupling_vel. `links_idx` are global link indices.
    """
    cfrc = scene.rigid_solver.dyn_state.links.cfrc_coupling_vel.to_numpy()
    forces = -cfrc[links_idx]               # (num_links, num_envs, 3)
    return np.transpose(forces, (1, 0, 2))  # (num_envs, num_links, 3)


def particle_counts(mpm_cfg, entities):
    """{patch name: particle count}, in config order."""
    return {p["name"]: int(e.n_particles) for p, e in zip(mpm_cfg["patches"], entities)}


class MudFootForces:
    """Mixin giving an env with self.scene and self.robot a get_foot_forces() over MPM mud."""

    _calf_idx = None  # global link indices of the foot-bearing links, filled on first call
    foot_names = None

    def get_foot_forces(self):
        """Per-foot reaction force from the mud, world frame, shape (num_envs, num_feet, 3).

        The foot sphere is fixed to the calf link, so the *_calf links carry the foot forces
        (see coupling_force).
        """
        if self._calf_idx is None:
            calves = [link for link in self.robot.links if link.name.endswith("_calf")]
            self._calf_idx = [link.idx for link in calves]  # global indices into the solver's link block
            self.foot_names = [link.name.replace("_calf", " foot") for link in calves]  # FL/FR/RL/RR
        return coupling_force(self.scene, self._calf_idx)
