import argparse
import os

import genesis as gs
'''
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rho : float, optional
        The rest density (kg/m³). Default is 1000.0.
    stiffness : float, optional
        State stiffness (N/m²). Controls how pressure increases with compression. Default is 50000.0.
    exponent : float, optional
        State exponent. Controls how nonlinearly pressure scales with density. Default is 7.0.
    mu : float, optional
        The viscosity of the liquid. Default is 0.005.
    gamma : float, optional
        The surface tension of the liquid. Default is 0.01.
    sampler : str, optional
        Particle sampler. Defaults to 'regular' for numerical stability with SPH.
    """


    rho: PositiveFloat = 1000.0
    stiffness: PositiveFloat = 50000.0
    exponent: PositiveFloat = 7.0
    mu: NonNegativeFloat = 0.005
    gamma: NonNegativeFloat = 0.01
    sampler: SamplerType = "regular"

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(precision="32", logging_level="info")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 2.4),
        ),
        vis_options=gs.options.VisOptions(
            visualize_sph_boundary=True,
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -3.15, 2.42),
            camera_lookat=(0.5, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    water = scene.add_entity(
        material=gs.materials.SPH.Liquid(
            rho=10000.0, 
            stiffness=50000.0, 
            exponent=8.0,
            mu=0.05, 
            gamma=0.01, 
            sampler="regular"
        ),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.6),
            size=(0.9, 1.6, 1.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
    )

    cube = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 2.4),# m
            size=(0.1, 0.1, 0.1), # m
            euler=(0, 0, 0), # Angle 
            fixed=False,
        ),
    )

    ########################## build ##########################
    scene.build()

    horizon = 500 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
