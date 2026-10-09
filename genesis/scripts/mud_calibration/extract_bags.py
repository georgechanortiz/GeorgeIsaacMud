import sys, numpy as np
from pathlib import Path
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
src, out = Path(sys.argv[1]), Path(sys.argv[2])
for bag in sorted(src.glob("*/*.mcap")):
    d = {"ls_t": [], "q": [], "dq": [], "tau": [], "quat": [], "sm_t": [], "vel": [], "pos": [], "ff": [], "fpb": [], "lc_t": [], "q_cmd": [], "kp": []}
    with open(bag, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, ch, msg, m in r.iter_decoded_messages(topics=["/lowstate", "/sportmodestate", "/lowcmd"]):
            t = msg.log_time * 1e-9
            if ch.topic == "/lowstate":
                ms = m.motor_state[:12]
                d["ls_t"].append(t); d["q"].append([x.q for x in ms]); d["dq"].append([x.dq for x in ms]); d["tau"].append([x.tau_est for x in ms])
                d["quat"].append(list(m.imu_state.quaternion))  # w, x, y, z
            elif ch.topic == "/sportmodestate":
                d["sm_t"].append(t); d["vel"].append(list(m.velocity)); d["pos"].append(list(m.position)); d["ff"].append(list(m.foot_force)); d["fpb"].append(list(m.foot_position_body))
            else:
                mc = m.motor_cmd[:12]
                d["lc_t"].append(t); d["q_cmd"].append([x.q for x in mc]); d["kp"].append([x.kp for x in mc])
    arr = {k: np.asarray(v, dtype=float) for k, v in d.items()}
    name = bag.parent.name.split("_")[0]
    np.savez(out / f"{name}.npz", **arr)
    print(name, {k: v.shape for k, v in arr.items()})
