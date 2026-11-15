"""Read the A2 hardware bags (MCAP files of Unitree SDK ROS 2 messages) into numpy arrays.

Needs `pip install mcap mcap-ros2-support`; ROS itself is not required.
"""

from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

TOPICS = ("/lowstate", "/lowcmd", "/sportmodestate")


def find_bags(bag_dir):
    """Every <bag_dir>/<bag name>/*.mcap, sorted."""
    return sorted(Path(bag_dir).glob("*/*.mcap"))


def run_name(bag_path):
    """Short run name from the bag directory: "mudwalk2_a2_20261006_154303" -> "mudwalk2"."""
    return Path(bag_path).parent.name.split("_")[0]


def read_bag(path):
    """Decode one bag into float arrays, one row per message.

    Times are log-clock seconds. Joints are in Unitree motor order: FR, FL, RR, RL, each
    (hip, thigh, calf) -- the same order as configs/a2_flat.py's joint_names.

        /lowstate        ls_t, q, dq, tau (tau_est), quat (IMU, w x y z), rpy, gyro
        /lowcmd          lc_t, q_cmd, kp
        /sportmodestate  sm_t, pos (odometry, world frame), vel (body frame),
                         ff (foot_force), fpb (foot_position_body)
    """
    d = {k: [] for k in ("ls_t", "q", "dq", "tau", "quat", "rpy", "gyro",
                         "lc_t", "q_cmd", "kp",
                         "sm_t", "pos", "vel", "ff", "fpb")}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, msg, m in reader.iter_decoded_messages(topics=list(TOPICS)):
            t = msg.log_time * 1e-9
            if channel.topic == "/lowstate":
                motors = m.motor_state[:12]
                d["ls_t"].append(t)
                d["q"].append([x.q for x in motors])
                d["dq"].append([x.dq for x in motors])
                d["tau"].append([x.tau_est for x in motors])
                d["quat"].append(list(m.imu_state.quaternion))
                d["rpy"].append(list(m.imu_state.rpy))
                d["gyro"].append(list(m.imu_state.gyroscope))
            elif channel.topic == "/lowcmd":
                motors = m.motor_cmd[:12]
                d["lc_t"].append(t)
                d["q_cmd"].append([x.q for x in motors])
                d["kp"].append([x.kp for x in motors])
            else:
                d["sm_t"].append(t)
                d["pos"].append(list(m.position))
                d["vel"].append(list(m.velocity))
                d["ff"].append(list(m.foot_force))
                d["fpb"].append(list(m.foot_position_body))
    return {k: np.asarray(v, dtype=float) for k, v in d.items()}
