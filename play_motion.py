#!/usr/bin/env python3
"""Play a LAFAN-format Unitree G1 motion CSV in MuJoCo and render it to MP4.

Example:
  python play_motion.py --csv motion.csv \
      --model unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml \
      --out motion.mp4
"""
import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

# CSV joint order (columns 7..35), per Unitree's LAFAN1 retargeting dataset.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def load_csv(path):
    """Load the CSV, skipping a header row if there is one."""
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        print("Header row detected, skipping it.")
        return np.loadtxt(path, delimiter=",", skiprows=1)


def build_scene(model_path):
    """Write a small wrapper XML (floor, sky, light) next to the robot model."""
    d = os.path.dirname(os.path.abspath(model_path))
    scene = os.path.join(d, "_scene_motion.xml")
    with open(scene, "w") as f:
        f.write(f"""<mujoco model="g1_motion_scene">
  <include file="{os.path.basename(model_path)}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="motion_floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>""")
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True, help="G1 29-DoF MJCF from unitree_rl_gym")
    ap.add_argument("--scene", default=None, help="use this XML as-is instead of building a floor wrapper")
    ap.add_argument("--out", default="motion.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--inspect", action="store_true", help="print CSV sanity checks and exit")
    args = ap.parse_args()

    motion = load_csv(args.csv)
    n_frames, n_cols = motion.shape
    print(f"{n_frames} frames x {n_cols} columns -> {n_frames / args.fps:.1f} s at {args.fps} fps")
    if n_cols != 7 + len(JOINT_NAMES):
        raise SystemExit(f"Expected 36 columns (3 pos + 4 quat + 29 joints), got {n_cols}.")
    if n_frames / args.fps > 30:
        print("WARNING: motion is longer than 30 s, the task requires under 30 s.")

    scene_path = args.scene or build_scene(args.model)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Map each CSV joint name to its address in qpos, by name (not by assumed order).
    qadr = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"Joint '{name}' not found in model. Use the 29-DoF model.")
        qadr.append(model.jnt_qposadr[jid])
    qadr = np.array(qadr)

    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free:
        raise SystemExit("Model has no free joint for the root.")
    fadr = model.jnt_qposadr[free[0]]

    # Sanity checks that also help answer "how did you know what the columns mean?"
    quat = motion[:, 3:7]
    print("Quaternion norm (cols 3-6): min %.4f max %.4f (should be ~1)"
          % (np.linalg.norm(quat, axis=1).min(), np.linalg.norm(quat, axis=1).max()))
    print("Root position range (cols 0-2) min:", motion[:, 0:3].min(0).round(2),
          "max:", motion[:, 0:3].max(0).round(2))
    bad = 0
    for k, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        col = motion[:, 7 + k]
        outside = col.min() < lo - 1e-3 or col.max() > hi + 1e-3
        bad += outside
        print(f"  col {7 + k:2d} {name:30s} data [{col.min():6.2f}, {col.max():6.2f}] "
              f"limits [{lo:6.2f}, {hi:6.2f}] {'<-- OUTSIDE' if outside else ''}")
    print(f"{bad} joints outside model limits (many = wrong order or degrees instead of radians)")
    if args.inspect:
        return

    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, 140, -15

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for row in motion:
        data.qpos[fadr:fadr + 3] = row[0:3]
        q = row[3:7] / np.linalg.norm(row[3:7])   # CSV is x,y,z,w
        data.qpos[fadr + 3:fadr + 7] = q[[3, 0, 1, 2]]  # MuJoCo wants w,x,y,z
        data.qpos[qadr] = row[7:]
        mujoco.mj_forward(model, data)

        cam.lookat[:] = [row[0], row[1], 0.8]  # follow the robot
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
    writer.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
