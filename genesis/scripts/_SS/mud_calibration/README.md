# A2 mud calibration

How the default mud in `scripts/eval_mpm_a2.py` was chosen: a 4-inch viscous-liquid layer
(E=1e5, nu=0.35, rho=1500) on the floor. See that script's docstring for the result table.

1. `extract_bags.py <a2_mud/hardware_bags> <out_dir>` decodes the MCAP bags into npz
   (needs `pip install mcap mcap-ros2-support`; ROS is not required).
2. `hw_targets.py <out_dir> results/hw_targets.json` computes grass and mud walking metrics:
   speed, positive mechanical power, and mean / 95th-percentile |tau_est| per joint group.
3. `run_one.py '<json>' results/<name>.json` runs one sim rollout (flat or mud) and records the
   same metrics, plus how much weight the mud carries and how deep the feet go.
4. `sweep.sh sweep1.txt` runs a list of materials. `python score.py` ranks them by how well
   (sim mud - sim flat) matches (hardware mud - hardware grass), in units of the hardware grass
   value. The baseline is `results/flat_s60.json`: flat ground at the same 60 substeps.

The score uses speed and calf torque (front mean, front 95th percentile, rear mean): the calf is
the joint that carries the mud's load on both the robot and in the sim. Left out:

- front-thigh torque and mechanical power. On hardware both rise because pulling the foot out of
  the mud takes work, and Genesis' MPM coupling cannot pull on a foot, so no material reproduces
  them;
- hip and rear-thigh torque, whose change in mud is a gait adaptation with the opposite sign in
  sim and on hardware.

`liq_inv_rho100` is a near-massless control (it shows the mud's load comes from its mass) and is
not ranked.

`plot_hw_report.py <npz_dir>` draws each hardware run as `reports/hardware_bags/<run>/foot_forces.png`
and `torques_by_leg.png`, in the sim reports' format. The bags have no foot-force data, so foot
forces are estimated from joint torques and are valid in stance only (see the script's docstring).

`metrics.py` holds what steps 2 and 3 share: the joint-group torque/power metrics and the
walking-window detection. Bag decoding lives in `scripts/hardware/bags.py`.
