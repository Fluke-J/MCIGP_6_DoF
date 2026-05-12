import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import logging
import cv2
import pyrealsense2 as rs
import torch.utils.data
import gc
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as SciRotation

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results_f
from skimage.feature import peak_local_max
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from xarm.wrapper import XArmAPI
from method import *

logging.basicConfig(level=logging.INFO)
sys.path.append("..")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


STARTUP_POSE_1 = np.array([231.6, 0.0, 400.0, 180.0, 0.0, 0.0], dtype=np.float32)
STARTUP_POSE_2 = np.array([404.0, 0.0, 400.0, 180.0, 0.0, 0.0], dtype=np.float32)
HOME_VIEW_POSE = np.array([403.8, 5.2, 650.2, 180.0, 0.0, 0.0], dtype=np.float32)
DROP_APPROACH_POSE = np.array([413.0, -536.0, 300.0, 180.0, 0.0, 0.0], dtype=np.float32)
DROP_RELEASE_POSE = np.array([413.0, -536.0, 150.0, 180.0, 0.0, 0.0], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='/home/robot/MCIGP_6dof/MCIGP/GRconvnet_RGBD_epoch_40_iou_0.52',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument('--hand-eye-npz', type=str, default='cam2eef.npz',
                        help='Path to cam2eef.npz containing R and T')
    parser.add_argument('--invert-hand-eye', action='store_true', dest='invert_hand_eye',
                        help='Invert the loaded hand-eye transform before use')
    parser.add_argument('--no-invert-hand-eye', action='store_false', dest='invert_hand_eye',
                        help='Use the loaded hand-eye transform directly')
    parser.set_defaults(invert_hand_eye=False)
    parser.add_argument('--pregrasp-mm', type=float, default=60.0,
                        help='TCP pre-grasp retreat distance in millimeters')
    parser.add_argument('--lift-mm', type=float, default=80.0,
                        help='TCP lift distance in millimeters after closing the gripper')
    parser.add_argument('--travel-z-mm', type=float, default=400.0,
                        help='Safe world-Z height for travel after lifting away from the object')
    parser.add_argument('--tcp-speed', type=float, default=80.0,
                        help='Absolute Cartesian motion speed for 6DoF grasp moves')
    parser.add_argument('--tcp-acc', type=float, default=500.0,
                        help='Absolute Cartesian motion acceleration for 6DoF grasp moves')
    parser.add_argument('--tcp-rot-rx-deg', type=float, default=0.0,
                        help='Extra TCP alignment rotation around X in degrees')
    parser.add_argument('--tcp-rot-ry-deg', type=float, default=0.0,
                        help='Extra TCP alignment rotation around Y in degrees')
    parser.add_argument('--tcp-rot-rz-deg', type=float, default=90.0,
                        help='Extra TCP alignment rotation around Z in degrees')
    parser.add_argument('--offset-x-mm', type=float, default=17.0,
                        help='Base-frame X correction from quick_click calibration')
    parser.add_argument('--offset-y-mm', type=float, default=15.0,
                        help='Base-frame Y correction from quick_click calibration')
    parser.add_argument('--offset-z-mm', type=float, default=70.0,
                        help='Base-frame Z correction from quick_click calibration')
    parser.add_argument('--min-grasp-z-mm', type=float, default=190.0,
                        help='Do not command grasp targets below this base-frame Z height')
    parser.add_argument('--min-tcp-yaw-deg', type=float, default=-90.0,
                        help='Minimum allowed TCP yaw in degrees to avoid cable winding')
    parser.add_argument('--max-tcp-yaw-deg', type=float, default=90.0,
                        help='Maximum allowed TCP yaw in degrees to avoid cable winding')
    parser.add_argument('--use-normal-orientation', action='store_true',
                        help='Use small surface-normal-derived roll/pitch adjustments instead of fixed orientation')
    parser.add_argument('--max-roll-tilt-deg', type=float, default=10.0,
                        help='Maximum allowed roll deviation from the home grasp orientation')
    parser.add_argument('--max-pitch-tilt-deg', type=float, default=10.0,
                        help='Maximum allowed pitch deviation from the home grasp orientation')
    parser.add_argument('--max-yaw-delta-deg', type=float, default=95.0,
                        help='Maximum allowed yaw deviation from the home grasp orientation')
    parser.add_argument('--no-safe-yaw-only-fallback', action='store_false', dest='safe_yaw_only_fallback',
                        help='Disable the safer yaw-only fallback when the 6DoF orientation looks risky')
    parser.set_defaults(safe_yaw_only_fallback=True)
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute the grasp pose without moving the robot')
    parser.add_argument('--normal-alpha', type=float, default=0.5,
                        help='Blend factor between top-down (0) and surface normal (1)')
    parser.add_argument('--mva-gain', type=float, default=0.7,
                        help='Scale factor for MVA XY correction')
    parser.add_argument('--mva-max-step-mm', type=float, default=80.0,
                        help='Maximum XY distance for one MVA correction')
    parser.add_argument('--mva-deadband-mm', type=float, default=5.0,
                        help='Skip MVA correction when XY error is below this distance')
    parser.add_argument('--mva-speed', type=float, default=60.0,
                        help='Cartesian speed for MVA correction')
    parser.add_argument('--mva-acc', type=float, default=200.0,
                        help='Cartesian acceleration for MVA correction')

    args = parser.parse_args()
    return args


def resolve_local_path(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path

    candidate_paths = [
        os.path.abspath(path),
        os.path.join(SCRIPT_DIR, path),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f'Could not find "{path}". Checked: {", ".join(candidate_paths)}'
    )


def get_H_eef_cam(path='cam2eef.npz', invert_hand_eye=False):
    cam2eff = np.load(resolve_local_path(path))
    R = cam2eff['R']
    tvec = np.asarray(cam2eff['T'], dtype=np.float32).reshape(3)
    H = np.eye(4, dtype=np.float32)
    H[:3, :3] = np.asarray(R, dtype=np.float32).reshape(3, 3)
    # The 6DoF planner below works in millimeters, matching xArm absolute poses.
    if np.max(np.abs(tvec)) < 10.0:
        tvec = tvec * 1000.0
    H[:3, -1] = tvec
    if invert_hand_eye:
        H = np.linalg.inv(H).astype(np.float32)
    return H


def make_H(Rm, t):
    H = np.eye(4, dtype=np.float32)
    H[:3, :3] = np.asarray(Rm, dtype=np.float32).reshape(3, 3)
    H[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return H


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector")
    return v / n


def T_local(x=0.0, y=0.0, z=0.0):
    return make_H(np.eye(3, dtype=np.float32), [x, y, z])


def pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = [float(v) for v in pose[:6]]
    H = np.eye(4, dtype=np.float32)
    H[:3, 3] = np.array([x, y, z], dtype=np.float32)
    H[:3, :3] = SciRotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    return H


def H_to_pose_xyzrpy_deg(H):
    rpy = SciRotation.from_matrix(H[:3, :3]).as_euler('xyz', degrees=True)
    return np.array([H[0, 3], H[1, 3], H[2, 3], rpy[0], rpy[1], rpy[2]], dtype=np.float32)


def wrap_angle_deg(angle_deg):
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def closest_angle_deg(angle_deg, reference_deg):
    wrapped_delta = wrap_angle_deg(float(angle_deg) - float(reference_deg))
    return float(reference_deg) + wrapped_delta


def choose_parallel_jaw_yaw_deg(raw_yaw_deg, min_yaw_deg, max_yaw_deg, reference_yaw_deg):
    candidates = [
        closest_angle_deg(raw_yaw_deg, reference_yaw_deg),
        closest_angle_deg(raw_yaw_deg + 180.0, reference_yaw_deg),
        closest_angle_deg(raw_yaw_deg - 180.0, reference_yaw_deg),
    ]

    valid_candidates = [yaw for yaw in candidates if float(min_yaw_deg) <= yaw <= float(max_yaw_deg)]
    if valid_candidates:
        return min(valid_candidates, key=lambda yaw: abs(yaw - float(reference_yaw_deg)))

    clamped_candidates = [
        float(np.clip(yaw, float(min_yaw_deg), float(max_yaw_deg)))
        for yaw in candidates
    ]
    return min(clamped_candidates, key=lambda yaw: abs(yaw - float(reference_yaw_deg)))


def clamp_pose_yaw_deg(pose_deg, min_yaw_deg, max_yaw_deg, reference_yaw_deg):
    pose = np.asarray(pose_deg, dtype=np.float32).copy()
    yaw = choose_parallel_jaw_yaw_deg(pose[5], min_yaw_deg, max_yaw_deg, reference_yaw_deg)
    pose[5] = yaw
    return pose


def estimate_robot_rpy_from_normal(normal_cam, grasp_angle_rad, args):
    roll = float(HOME_VIEW_POSE[3])
    pitch = float(HOME_VIEW_POSE[4])
    yaw = float(HOME_VIEW_POSE[5]) + np.clip(
        -np.degrees(float(grasp_angle_rad)),
        -float(args.max_yaw_delta_deg),
        float(args.max_yaw_delta_deg),
    )
    mode = 'yaw_only'

    if normal_cam is not None and args.use_normal_orientation:
        nx, ny, nz = [float(v) for v in normal_cam]
        if abs(nz) > 1e-6:
            roll_delta = np.degrees(np.arctan2(ny, -nz))
            pitch_delta = -np.degrees(np.arctan2(nx, -nz))
            roll += np.clip(roll_delta, -float(args.max_roll_tilt_deg), float(args.max_roll_tilt_deg))
            pitch += np.clip(pitch_delta, -float(args.max_pitch_tilt_deg), float(args.max_pitch_tilt_deg))
            mode = 'tilted_6dof'

    roll = closest_angle_deg(roll, HOME_VIEW_POSE[3])
    pitch = closest_angle_deg(pitch, HOME_VIEW_POSE[4])
    yaw = closest_angle_deg(yaw, HOME_VIEW_POSE[5])
    return roll, pitch, yaw, mode


def deproject_pixel_to_cam_mm(color_intrinsics, u, v, depth_m):
    point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [float(u), float(v)], float(depth_m))
    return np.asarray(point, dtype=np.float32) * 1000.0


def estimate_surface_normal_cam(depth_img, color_intrinsics, u, v, radius=4,
                                depth_min_m=0.08, depth_max_m=2.0):
    if depth_img.ndim == 3:
        depth_img = np.squeeze(depth_img, axis=-1)

    h, w = depth_img.shape
    points = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu = u + du
            vv = v + dv
            if uu < 0 or uu >= w or vv < 0 or vv >= h:
                continue
            depth_value = float(depth_img[vv, uu])
            if not np.isfinite(depth_value) or depth_value < depth_min_m or depth_value > depth_max_m:
                continue
            points.append(deproject_pixel_to_cam_mm(color_intrinsics, uu, vv, depth_value))

    if len(points) < 6:
        raise RuntimeError('Too few valid depth points for surface normal estimation')

    points = np.asarray(points, dtype=np.float32)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = normalize(vh[-1])

    if normal[2] > 0:
        normal = -normal
    return normal


def build_H_cam_grasp(depth_img, color_intrinsics, u, v, depth_m, grasp_angle_rad, args):
    point_cam_mm = deproject_pixel_to_cam_mm(color_intrinsics, u, v, depth_m)

    # ===== NORMAL ESTIMATION =====
    try:
        normal_cam = estimate_surface_normal_cam(depth_img, color_intrinsics, u, v)
    except Exception:
        normal_cam = None

    # ===== BLENDED 6DOF (KEY FIX) =====
    z_top = np.array([0, 0, -1], dtype=np.float32)

    if normal_cam is not None and args.use_normal_orientation:
        z_normal = normalize(-normal_cam)

        # 🔥 blend factor
        alpha = float(np.clip(args.normal_alpha, 0.0, 1.0))

        z_axis = normalize((1 - alpha) * z_top + alpha * z_normal)
    else:
        z_axis = z_top

    # ===== GRIPPER X AXIS (from grasp angle) =====
    x_hint = np.array([
        np.cos(grasp_angle_rad),
        np.sin(grasp_angle_rad),
        0.0
    ], dtype=np.float32)

    x_axis = x_hint - np.dot(x_hint, z_axis) * z_axis

    if np.linalg.norm(x_axis) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(fallback, z_axis)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = fallback - np.dot(fallback, z_axis) * z_axis

    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R_cam_tcp = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    H_cam_tcp = make_H(R_cam_tcp, point_cam_mm)

    return H_cam_tcp, point_cam_mm, normal_cam


def compute_6dof_target_poses(arm, depth_img, color_intrinsics, pixel_xy, depth_m,
                              grasp_angle_rad, hand_eye_transform, args,
                              pregrasp_offset_mm=60.0, lift_offset_mm=80.0):
    ret, eff_pose = arm.get_position(is_radian=False)
    if ret != 0:
        print("Error reading robot pose")
        return None

    H_base_eef = pose_to_matrix(eff_pose)
    H_cam_tcp, point_cam_mm, normal_cam = build_H_cam_grasp(
        depth_img,
        color_intrinsics,
        int(pixel_xy[0]),
        int(pixel_xy[1]),
        float(depth_m),
        float(grasp_angle_rad),
        args
    )
    
    R_fix = SciRotation.from_euler(
        'xyz',
        [
            float(args.tcp_rot_rx_deg),
            float(args.tcp_rot_ry_deg),
            float(args.tcp_rot_rz_deg),
        ],
        degrees=True,
    ).as_matrix().astype(np.float32)

    H_fix = np.eye(4, dtype=np.float32)
    H_fix[:3, :3] = R_fix

    H_base_tcp = H_base_eef @ hand_eye_transform @ H_cam_tcp @ H_fix
    H_base_tcp[0, 3] += float(args.offset_x_mm)
    H_base_tcp[1, 3] += float(args.offset_y_mm)
    raw_target_z = float(H_base_tcp[2, 3] + float(args.offset_z_mm))
    clamped_target_z = max(raw_target_z, float(args.min_grasp_z_mm))
    H_base_tcp[2, 3] = clamped_target_z

    pose_target = H_to_pose_xyzrpy_deg(H_base_tcp)
    # 🔥 LIMIT orientation jump
    max_angle_step = 20  # degrees

    for i in range(3, 6):  # roll, pitch, yaw
        delta = pose_target[i] - HOME_VIEW_POSE[i]
        delta = np.clip(delta, -max_angle_step, max_angle_step)
        pose_target[i] = HOME_VIEW_POSE[i] + delta

    
    raw_target_yaw = float(closest_angle_deg(pose_target[5], HOME_VIEW_POSE[5]))
    pose_target = clamp_pose_yaw_deg(
        pose_target,
        min_yaw_deg=args.min_tcp_yaw_deg,
        max_yaw_deg=args.max_tcp_yaw_deg,
        reference_yaw_deg=HOME_VIEW_POSE[5],
    )
    pose_pre = pose_target.copy()
    pose_lift = pose_target.copy()
    pose_pre[2] = pose_target[2] + float(pregrasp_offset_mm)
    pose_lift[2] = pose_target[2] + float(lift_offset_mm)

    pose_travel = pose_lift.copy()
    pose_travel[2] = max(float(pose_lift[2]), float(args.travel_z_mm))

    pose_reorient = pose_travel.copy()
    pose_reorient[3:] = HOME_VIEW_POSE[3:]

    return {
        'pose_target': pose_target,
        'pose_pre': pose_pre,
        'pose_lift': pose_lift,
        'pose_travel': pose_travel,
        'pose_reorient': pose_reorient,
        'point_cam_mm': point_cam_mm,
        'normal_cam': normal_cam,
        'orientation_mode': 'full_6dof',
        'raw_target_z_mm': raw_target_z,
        'clamped_target_z_mm': clamped_target_z,
        'raw_target_yaw_deg': raw_target_yaw,
        'clamped_target_yaw_deg': float(pose_target[5]),
    }


def move_abs_pose(arm, pose_deg, speed=80, mvacc=500, wait=True):
    pose_deg = np.asarray(pose_deg, dtype=np.float32).reshape(6)
    
    # reduce jerk by lowering speed/acc
    speed = min(speed, 80)
    mvacc = min(mvacc, 250)
    
    code = arm.set_position(
        x=float(pose_deg[0]),
        y=float(pose_deg[1]),
        z=float(pose_deg[2]),
        roll=float(pose_deg[3]),
        pitch=float(pose_deg[4]),
        yaw=float(pose_deg[5]),
        speed=float(speed),
        mvacc=float(mvacc),
        wait=wait,
        is_radian=False,
    )
    if code not in (0, None):
        raise RuntimeError(f'arm.set_position failed, code={code}, pose={pose_deg.tolist()}')


def wait_until_pose(arm, pose_deg, tolerance=3.0):
    target = np.asarray(pose_deg, dtype=np.float32).reshape(6)
    while True:
        ret, current_pose = arm.get_position(is_radian=False)
        if ret == 0 and all(abs(float(current_pose[i]) - float(target[i])) < tolerance for i in range(3)):
            break
        time.sleep(0.1)


def execute_mva_xy_move(arm, hand_eye_transform, point_cam_m, args, label='MVA'):
    point_cam_m = np.asarray(point_cam_m, dtype=np.float32).reshape(3)
    delta_cam_mm = np.array([point_cam_m[0], point_cam_m[1], 0.0], dtype=np.float32) * 1000.0

    ret, current_pose = arm.get_position(is_radian=False)
    if ret != 0:
        print(f'[{label}] Cannot read robot pose; skip MVA')
        return False

    H_base_eef = pose_to_matrix(current_pose)
    R_base_cam = H_base_eef[:3, :3] @ hand_eye_transform[:3, :3]
    delta_base_mm = R_base_cam @ delta_cam_mm
    delta_xy = np.asarray(delta_base_mm[:2], dtype=np.float32) * float(args.mva_gain)

    step_norm = float(np.linalg.norm(delta_xy))
    if step_norm < float(args.mva_deadband_mm):
        print(f'[{label}] MVA shift inside deadband:', np.round(delta_xy, 2), 'mm')
        return False

    max_step = float(args.mva_max_step_mm)
    if step_norm > max_step:
        delta_xy *= max_step / step_norm

    target_pose = np.asarray(current_pose, dtype=np.float32).copy()
    target_pose[0] += float(delta_xy[0])
    target_pose[1] += float(delta_xy[1])

    print(f'[{label}] Camera point m:', np.round(point_cam_m, 4))
    print(f'[{label}] Base XY correction mm:', np.round(delta_xy, 2))

    if args.dry_run:
        print(f'[{label}] Dry run: MVA move skipped')
        return False

    move_abs_pose(
        arm,
        target_pose,
        speed=args.mva_speed,
        mvacc=args.mva_acc,
        wait=True,
    )
    wait_until_pose(arm, target_pose)
    return True

"""
def execute_grasp_6dof(arm, args, depth_img, color_intrinsics, pixel_xy, depth_m,
                       grasp_angle_rad, grasp_width, hand_eye_transform):
    try:
        result = compute_6dof_target_poses(
            arm=arm,
            depth_img=depth_img,
            color_intrinsics=color_intrinsics,
            pixel_xy=pixel_xy,
            depth_m=depth_m,
            grasp_angle_rad=grasp_angle_rad,
            hand_eye_transform=hand_eye_transform,
            args=args,
            pregrasp_offset_mm=args.pregrasp_mm,
            lift_offset_mm=args.lift_mm,
        )
    except Exception as exc:
        print(f"6DoF grasp planning failed: {exc}")
        return
    if result is None:
        return

    print("\n===== 6DOF DEBUG =====")
    print("Pixel XY:", [int(pixel_xy[0]), int(pixel_xy[1])])
    print("Depth (m):", float(depth_m))
    print("Hand-eye translation (mm):", np.round(hand_eye_transform[:3, 3], 3))
    print("Hand-eye invert:", bool(args.invert_hand_eye))
    print("TCP rot xyz (deg):", [float(args.tcp_rot_rx_deg), float(args.tcp_rot_ry_deg), float(args.tcp_rot_rz_deg)])
    print("Base-frame offset (mm):", [float(args.offset_x_mm), float(args.offset_y_mm), float(args.offset_z_mm)])
    print("Target Z raw/clamped (mm):", round(float(result['raw_target_z_mm']), 3), round(float(result['clamped_target_z_mm']), 3))
    print("Target yaw raw/clamped (deg):", round(float(result['raw_target_yaw_deg']), 3), round(float(result['clamped_target_yaw_deg']), 3))
    print("Yaw safe range (deg):", [float(args.min_tcp_yaw_deg), float(args.max_tcp_yaw_deg)])
    print("Point cam (mm):", np.round(result['point_cam_mm'], 3))
    print("Surface normal cam:", None if result['normal_cam'] is None else np.round(result['normal_cam'], 4))
    print("Orientation mode:", result['orientation_mode'])
    print("Pre-grasp pose:", np.round(result['pose_pre'], 3))
    print("Target pose:", np.round(result['pose_target'], 3))
    print("Lift pose:", np.round(result['pose_lift'], 3))
    print("Travel pose:", np.round(result['pose_travel'], 3))
    print("Reorient pose:", np.round(result['pose_reorient'], 3))
    print("======================\n")

    if args.dry_run:
        return

    arm.set_gripper_position(float(grasp_width) * 7.5, wait=True)
    move_abs_pose(arm, result['pose_pre'], speed=args.tcp_speed, mvacc=args.tcp_acc, wait=True)
    move_abs_pose(arm, result['pose_target'], speed=min(40.0, float(args.tcp_speed)), mvacc=args.tcp_acc, wait=True)
    arm.set_gripper_mode(0)
    arm.set_gripper_position(20, wait=True)
    move_abs_pose(arm, result['pose_lift'], speed=args.tcp_speed, mvacc=args.tcp_acc, wait=True)
    move_abs_pose(arm, result['pose_travel'], speed=args.tcp_speed, mvacc=args.tcp_acc, wait=True)
    move_abs_pose(arm, result['pose_reorient'], speed=args.tcp_speed, mvacc=args.tcp_acc, wait=True)
    move_abs_pose(arm, DROP_APPROACH_POSE, speed=200, mvacc=args.tcp_acc, wait=True)
    move_abs_pose(arm, DROP_RELEASE_POSE, speed=200, mvacc=args.tcp_acc, wait=True)
    arm.set_gripper_position(850, wait=True)
    move_abs_pose(arm, DROP_APPROACH_POSE, speed=200, mvacc=args.tcp_acc, wait=True)
    move_abs_pose(arm, HOME_VIEW_POSE, speed=200, mvacc=args.tcp_acc, wait=True)
    wait_until_pose(arm, HOME_VIEW_POSE)
"""

def execute_grasp_6dof(arm, args, depth_img, color_intrinsics, pixel_xy, depth_m,
                       grasp_angle_rad, grasp_width, hand_eye_transform):

    # 🔥 disable unstable tilt for now
    args.use_normal_orientation = False

    try:
        result = compute_6dof_target_poses(
            arm=arm,
            depth_img=depth_img,
            color_intrinsics=color_intrinsics,
            pixel_xy=pixel_xy,
            depth_m=depth_m,
            grasp_angle_rad=grasp_angle_rad,
            hand_eye_transform=hand_eye_transform,
            args=args,
            pregrasp_offset_mm=args.pregrasp_mm,
            lift_offset_mm=args.lift_mm,
        )
    except Exception as exc:
        print(f"6DoF grasp planning failed: {exc}")
        return

    if result is None:
        return

    print("\n===== SAFE EXECUTION =====")
    print("Target pose:", np.round(result['pose_target'], 2))
    print("=========================\n")

    if args.dry_run:
        return

    # =========================
    # SAFE MOTION SEQUENCE
    # =========================

    # Open gripper
    arm.set_gripper_position(float(grasp_width) * 7.5, wait=True)

    # 🔥 MID WAYPOINT (VERY IMPORTANT)
    mid_pose = result['pose_pre'].copy()
    mid_pose[2] = result['pose_target'][2] + 60

    move_abs_pose(arm, mid_pose, speed=60)
    time.sleep(0.1)

    # Move above object
    move_abs_pose(arm, result['pose_pre'], speed=60)
    time.sleep(0.1)

    # 🔥 slow approach
    move_abs_pose(arm, result['pose_target'], speed=20)
    time.sleep(0.1)

    # Close gripper
    arm.set_gripper_position(20, wait=True)

    # Lift
    move_abs_pose(arm, result['pose_lift'], speed=60)
    time.sleep(0.1)

    # Travel
    move_abs_pose(arm, result['pose_travel'], speed=80)
    time.sleep(0.1)

    # Reorient
    move_abs_pose(arm, result['pose_reorient'], speed=80)
    time.sleep(0.1)

    # Drop
    move_abs_pose(arm, DROP_APPROACH_POSE, speed=120)
    move_abs_pose(arm, DROP_RELEASE_POSE, speed=60)

    arm.set_gripper_position(850, wait=True)

    move_abs_pose(arm, DROP_APPROACH_POSE, speed=120)
    move_abs_pose(arm, HOME_VIEW_POSE, speed=120)

    wait_until_pose(arm, HOME_VIEW_POSE)

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.empty_cache()
    hand_eye_transform = get_H_eef_cam(args.hand_eye_npz, invert_hand_eye=args.invert_hand_eye)

    # Initialize the robot and gripper
    ip = '192.168.1.232'
    arm = XArmAPI(ip, is_radian=True)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    move_abs_pose(arm, STARTUP_POSE_1, speed=200, mvacc=args.tcp_acc, wait=True)
    move_abs_pose(arm, STARTUP_POSE_2, speed=200, mvacc=args.tcp_acc, wait=True)

    code = arm.set_gripper_enable(True)
    code = arm.set_gripper_mode(0)

# ----------------------------------------------------------------------------------------------------------------------------#
                                                   # MVA part
# ----------------------------------------------------------------------------------------------------------------------------#

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id='943222070907')
    color_intrinsics = cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu = torch.cuda.current_device()
    device = get_device(gpu)

    try:
        fig = plt.figure(figsize=(10, 10))
        is_grasping = True  # # Indicate whether crawling is in progress
        exit_requested = False  # Indicate whether to request to exit
        count = 0  # Initialize the buffer frame counter
        move = 0  # Initialize the center movement frame counter
        rotation = 0  # Initialize the image rotation angle

        while True:
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            aligned_depth_frame = image_bundle['aligned_depth_frame']

            # Buffer frames to prevent poor prediction
            if count < 24:
                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                with torch.no_grad():
                    xc = x.to(device)
                    pred = net.predict(xc)
                    count += 1
                    if count == 1:
                        move_abs_pose(arm, HOME_VIEW_POSE, speed=200, mvacc=args.tcp_acc, wait=True)
                        wait_until_pose(arm, HOME_VIEW_POSE)
                        time.sleep(2)
                        print("Entering the initial view, ready to grasping...")

            # MVA!!!
            elif move < 3:
                move += 1

                # Gobal MVA
                if move == 1:
                    center_x, center_y = depth.shape[1] // 2, depth.shape[0] // 2
                    start_x, start_y = center_x - 310, center_y - 160
                    end_x, end_y = center_x + 310, center_y + 160
                    center_region = depth[start_y:end_y, start_x:end_x]

                    filtered_depth_img = np.where(center_region < 0.2, np.inf, center_region)
                    min_depth_value = np.min(filtered_depth_img)
                    min_index = np.argmin(filtered_depth_img)
                    min_coords = np.unravel_index(min_index, filtered_depth_img.shape)

                    # Move camera
                    if min_depth_value < 0.662:  # Prevent collision with table
                        min_coords_squeezed = (min_coords[0] + start_y, min_coords[1] + start_x)

                        dis = depth[min_coords_squeezed[0], min_coords_squeezed[1]]
                        x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                  pixel=[min_coords_squeezed[1], min_coords_squeezed[0]],
                                                                  depth=dis)
                        campos = [x, y, z]

                        execute_mva_xy_move(
                            arm=arm,
                            hand_eye_transform=hand_eye_transform,
                            point_cam_m=campos,
                            args=args,
                            label='Global MVA',
                        )

                        time.sleep(2)

                # Local MVA
                else:
                    # Dynamic Monozone
                    center_x, center_y = depth.shape[1] // 2, depth.shape[0] // 2
                    start_x, start_y = center_x - 112, center_y - 112
                    end_x, end_y = center_x + 112, center_y + 112
                    center_region = depth[start_y:end_y, start_x:end_x]

                    filtered_depth_img = np.where(center_region < 0.2, np.inf, center_region)
                    min_depth_value = np.min(filtered_depth_img)
                    min_index = np.argmin(filtered_depth_img)
                    min_coords = np.unravel_index(min_index, filtered_depth_img.shape)

                    if min_depth_value < 0.662:
                        min_coords_squeezed = (min_coords[0] + start_y, min_coords[1] + start_x)

                        dis = depth[min_coords_squeezed[0], min_coords_squeezed[1]]
                        x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                  pixel=[min_coords_squeezed[1],
                                                                         min_coords_squeezed[0]],
                                                                  depth=dis)
                        campos = [x, y, z]

                        # move robot
                        execute_mva_xy_move(
                            arm=arm,
                            hand_eye_transform=hand_eye_transform,
                            point_cam_m=campos,
                            args=args,
                            label='Local MVA',
                        )

                        time.sleep(0.2)

# ----------------------------------------------------------------------------------------------------------------------------#
                                                       # ISGD part
# ----------------------------------------------------------------------------------------------------------------------------#

            # Conduct CPS in the aligned view
            else:

                if min_depth_value < 0.1:
                    print('Dangerous grasp, jump to next frame:', min_depth_value)
                    count = 0
                    move = 0
                    continue

                elif min_depth_value > 0.662:
                    print('Grasp finised')
                    count = 0
                    move = 0
                    time.sleep(2)
                    break

                else:
                    # initialize SAM
                    depth_squeezed = np.squeeze(depth, axis=2)
                    initial_point = (240, 320)
                    depth_value = min_depth_value
                    delta_d = 0.008
                    structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    d_mask = minkowski_sum(depth_squeezed, initial_point, depth_value, delta_d, structure_element)
                    min_coords_squeezed = (240, 320)

                    sam_checkpoint = "/home/robot/MCIGP_6dof/MCIGP/sam_vit_b_01ec64.pth"
                    model_type = "vit_b"
                    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    sam.to(device=device)
                    predictor = SamPredictor(sam)
                    predictor.set_image(rgb)

                    # Single-point segmentation
                    input_point = np.array([[min_coords_squeezed[1], min_coords_squeezed[0]]])
                    input_label = np.array([1])
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                    mask = masks.squeeze(0)
                    edges = sobel_compute(mask)

                    # Find CPS points
                    point1, point2 = find_farthest_points(edges)
                    midpoint, slope, x_values, y_values = find_perpendicular(point1, point2, (480, 640))

                    # Find intersection points (cross points)
                    intersections = find_intersection_points(x_values, y_values, edges)

                    # CPS segmentation
                    if len(intersections) < 2:
                        input_point = np.array([input_point[0], (point1[1], point1[0]), (point2[1], point2[0])])
                        input_label = np.array([1, 1, 1])
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=False,
                        )
                    else:
                        input_point = np.array(
                            [input_point[0], (point1[1], point1[0]), (point2[1], point2[0]),\
                             (intersections[0][0], intersections[0][1]),
                             (intersections[-1][0], intersections[-1][1])])
                        input_label = np.array([1, 1, 1, 1, 1])
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=False,
                        )

                    second_largest_mask = masks.squeeze(0)
                    second_largest_mask_i = second_largest_mask.astype(int)
                    points = np.column_stack(np.nonzero(second_largest_mask_i))
                    print("mask points:", len(points))

                    if len(points) < 1500:
                        print("mask is too small:", len(points))
                        # Refine mask
                        second_largest_mask = second_largest_mask_i + d_mask
                        second_largest_mask = second_largest_mask.astype(bool)

                    rgb[~second_largest_mask] = [255, 255, 255]

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                        # Grasp detection
                        x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                        with torch.no_grad():
                            xc = x.to(device)
                            pred = net.predict(xc)
                            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'],
                                                                            pred['width'])

                        grasp_point1 = peak_local_max(q_img, min_distance=1, threshold_abs=0.3, num_peaks=100)
                        print(len(grasp_point1))

                        edges = sobel_compute(second_largest_mask)

                        # The first part of GCO
                        for grasp_point in grasp_point1:
                            length = width_img[grasp_point[0], grasp_point[1]]

                            # Calculate the optimal angle and minimum angle difference
                            best_angle, min_difference = calculate_total_difference(grasp_point, length,
                                                                                    edges)
                            # Update ang_img
                            if best_angle is not None:
                                ang_img[grasp_point[0], grasp_point[1]] = best_angle

                        graspable_points = grasp_point1

                        # The second part of GCO
                        graspable_points_adjacent = []
                        for grasp_point in graspable_points:
                            angle = ang_img[grasp_point[0], grasp_point[1]]
                            length = width_img[grasp_point[0], grasp_point[1]]
                            width = 43

                            # decode grasps
                            p1, p2, p3, p4 = decode_box(angle, grasp_point, length, width)

                            center, intersection, max_side_length = get_intersection_and_rectangle(p1, p2, p3, p4,
                                                                                                   second_largest_mask)
                            if center is not None:

                                all_relations_true = True

                                for i in range(13, 14):
                                    f_grasp_point = [int(center[1]) - 128, int(center[0]) - 208]
                                    f_length = min(max_side_length + i, 113)
                                    f_width = 43
                                    f_angle = angle

                                    p1, p2, p3, p4 = decode_box(f_angle, f_grasp_point, f_length, f_width)

                                    relation_one = compute_adjacent_relation(p1, p2, p3, p4, second_largest_mask,
                                                                             f_grasp_point, depth)

                                    if not relation_one:
                                        all_relations_true = False
                                        break

                                if all_relations_true:
                                    graspable_points_adjacent.append(grasp_point)

                        if len(graspable_points_adjacent) == 0:
                            print('no graspable_points，preparing to rotate image')

                            for i in np.linspace(30, 300, 5):
                                rotation = i
                                depth_r = np.mean(depth, axis=2)

                                # rotate rgb and depth
                                pil_rgb = Image.fromarray(rgb.astype('uint8'), 'RGB')
                                pil_depth = Image.fromarray(depth_r)
                                rotated_rgb = pil_rgb.rotate(rotation, resample=Image.BILINEAR)
                                rotated_depth = pil_depth.rotate(rotation, resample=Image.BILINEAR)
                                rotated_rgb = np.array(rotated_rgb)
                                rotated_depth_np = np.array(rotated_depth)
                                rotated_depth_np = np.stack([rotated_depth_np] * 1, axis=-1)

                                # rotate mask
                                r_second_largest_mask = second_largest_mask
                                pil_mask = Image.fromarray(r_second_largest_mask.astype(np.uint8))
                                r_second_largest_mask = pil_mask.rotate(rotation, resample=Image.BILINEAR)
                                r_second_largest_mask = np.array(r_second_largest_mask)

                                # Recalculate the edge of the new mask
                                r_edges = sobel_compute(r_second_largest_mask)

                                r_x, r_depth_img, r_rgb_img = cam_data.get_data(rgb=rotated_rgb, depth=rotated_depth_np)
                                with torch.no_grad():
                                    r_xc = r_x.to(device)
                                    r_pred = net.predict(r_xc)
                                    r_q_img, r_ang_img, r_width_img = post_process_output(r_pred['pos'], r_pred['cos'], r_pred['sin'],
                                                                                    r_pred['width'])

                                    grasp_point1 = peak_local_max(r_q_img, min_distance=1, threshold_abs=0.1, num_peaks=100)

                                    for grasp_point in grasp_point1:
                                        length = r_width_img[grasp_point[0], grasp_point[1]]

                                        best_angle, min_difference = calculate_total_difference(grasp_point, length,
                                                                                                r_edges)
                                        if best_angle is not None:
                                            r_ang_img[grasp_point[0], grasp_point[1]] = best_angle

                                graspable_points = grasp_point1

                                # Filter grasps that collide with adjacent objects
                                graspable_points_adjacent = []
                                for grasp_point in graspable_points:
                                    angle = r_ang_img[grasp_point[0], grasp_point[1]]
                                    length = r_width_img[grasp_point[0], grasp_point[1]]
                                    width = 43

                                    # decode grasps
                                    p1, p2, p3, p4 = decode_box(angle, grasp_point, length, width)

                                    center, intersection, max_side_length = get_intersection_and_rectangle(p1, p2, p3,
                                                                                                           p4,
                                                                                                           r_second_largest_mask)
                                    if center is not None:

                                        all_relations_true = True

                                        for i in range(13, 14):
                                            f_grasp_point = [int(center[1]) - 128, int(center[0]) - 208]
                                            f_length = min(max_side_length + i, 113)
                                            f_width = 43
                                            f_angle = angle

                                            p1, p2, p3, p4 = decode_box(f_angle, f_grasp_point, f_length, f_width)

                                            relation_one = compute_adjacent_relation(p1, p2, p3, p4,
                                                                                     r_second_largest_mask,
                                                                                     f_grasp_point, rotated_depth_np)

                                            if not relation_one:
                                                all_relations_true = False
                                                break

                                        if all_relations_true:
                                            graspable_points_adjacent.append(grasp_point)

                                if len(graspable_points_adjacent) != 0:
                                    print("can grasp now, rotated angle:", rotation)
                                    break

                            if len(graspable_points_adjacent) == 0:
                                print('still can not grasp this object after rotated, jump to next frame')
                                plot_results(fig=fig,
                                             rgb_img=cam_data.get_rgb(rotated_rgb, False),
                                             depth_img=np.squeeze(cam_data.get_depth(rotated_depth_np)),
                                             grasp_q_img=r_q_img,
                                             grasp_angle_img=r_ang_img,
                                             no_grasps=args.n_grasps,
                                             grasp_width_img=r_width_img,
                                             point=None,
                                             fine_point=None,
                                             fine_width=None)
                                count = 0
                                move = 0
                                rotation = 0
                                continue

                        # Update variables
                        if rotation != 0:
                            rgb = rotated_rgb
                            depth = rotated_depth_np
                            q_img = r_q_img
                            width_img = r_width_img
                            ang_img = r_ang_img
                            second_largest_mask = r_second_largest_mask
                            edges = r_edges
                            depth_img = r_depth_img

                        # Select the best grasp based on depth
                        graspable_points_depth = []

                        for grasp_point in graspable_points_adjacent:
                            x, y = grasp_point[0], grasp_point[1]
                            depth_img_squeezed = np.squeeze(depth_img, axis=0)
                            depth_value = depth_img_squeezed[x, y]
                            graspable_points_depth.append((grasp_point, depth_value))

                        if graspable_points_depth:
                            graspable_points_depth.sort(key=lambda x: x[1])
                            best_grasp_point = graspable_points_depth[0][0]
                        else:
                            print("No graspable points found.")
                            count = 0
                            move = 0
                            rotation = 0
                            continue

                        # Refine optimal grasp
                        F_best_grasp_point = None
                        for width in range(10, 2, -1):
                            angle = ang_img[best_grasp_point[0], best_grasp_point[1]]
                            length = width_img[best_grasp_point[0], best_grasp_point[1]]

                            p1, p2, p3, p4 = decode_box(angle, best_grasp_point, length, width)

                            center, intersection, max_side_length = get_intersection_and_rectangle(p1, p2, p3, p4,
                                                                                                   second_largest_mask)
                            if center is not None:
                                F_best_grasp_point = [center[1], center[0]]
                                F_length = min(max_side_length+13, 113)
                                F_angle = ang_img[best_grasp_point[0], best_grasp_point[1]]
                                break

                            if F_best_grasp_point is None:
                                print('No intersection, jump to next frame:')
                                count = 0
                                move = 0
                                rotation = 0

                                plot_results(fig=fig,
                                             rgb_img=cam_data.get_rgb(rgb, False),
                                             depth_img=np.squeeze(cam_data.get_depth(depth)),
                                             grasp_q_img=q_img,
                                             grasp_angle_img=ang_img,
                                             no_grasps=args.n_grasps,
                                             grasp_width_img=width_img,
                                             point=best_grasp_point,
                                             fine_point=None,
                                             fine_width=None)
                                continue

                        C_F_best_grasp_point = [F_best_grasp_point[1] - 128, F_best_grasp_point[0] - 208]
                        print("Finetuned Pose (center, length, angle):", F_best_grasp_point, F_length, F_angle)

                        center = [F_best_grasp_point[0], F_best_grasp_point[1]]
                        data_array = [int(center[0]), int(center[1])]  # Project 224*224 back to 640*480
                        print("rotated_data_array:", data_array)

                        # Project angle back to aligned view
                        if rotation != 0:
                            rotated_x = data_array[0]
                            rotated_y = data_array[1]
                            original_width = 640
                            original_height = 480

                            center_x = original_width / 2
                            center_y = original_height / 2

                            # Calculate the rotation matrix
                            theta = np.radians(-rotation)
                            cos_theta = np.cos(theta)
                            sin_theta = np.sin(theta)
                            rotation_matrix = np.array([[cos_theta, -sin_theta],
                                                        [sin_theta, cos_theta]])

                            # Calculate the offset of the rotated pixel coordinates
                            rotated_offset_x = rotated_x - center_x
                            rotated_offset_y = rotated_y - center_y

                            # Calculate the pixel coordinates (x, y) before rotation
                            original_offset_x, original_offset_y = np.dot(rotation_matrix.T,
                                                                          [rotated_offset_x, rotated_offset_y])

                            # Convert the offset back to the coordinates (x, y) of the original image
                            original_x = original_offset_x + center_x
                            original_y = original_offset_y + center_y

                            original_x = int(round(original_x))
                            original_y = int(round(original_y))

                            data_array[0] = original_x
                            data_array[1] = original_y

                        dis = float(np.asarray(depth[data_array[1], data_array[0]]).squeeze())
                        print("dis:", dis)
                        x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                  pixel=[data_array[0], data_array[1]],
                                                                  depth=dis)
                        campos = [x, y, z]
                        if dis < 0.1 or dis > 0.662:
                            print('Dangerous grasp, jump to next frame:', dis)
                            count = 0
                            move = 0
                            rotation = 0

                            plot_results(fig=fig,
                                         rgb_img=cam_data.get_rgb(rgb, False),
                                         depth_img=np.squeeze(cam_data.get_depth(depth)),
                                         grasp_q_img=q_img,
                                         grasp_angle_img=ang_img,
                                         no_grasps=args.n_grasps,
                                         grasp_width_img=width_img,
                                         point=best_grasp_point,
                                         fine_point=C_F_best_grasp_point,
                                         fine_width=F_length)
                            continue

                        plot_results(fig=fig,
                                     rgb_img=cam_data.get_rgb(rgb, False),
                                     depth_img=np.squeeze(cam_data.get_depth(depth)),
                                     grasp_q_img=q_img,
                                     grasp_angle_img=ang_img,
                                     no_grasps=args.n_grasps,
                                     grasp_width_img=width_img,
                                     point=best_grasp_point,
                                     fine_point=C_F_best_grasp_point,
                                     fine_width=F_length)

# ----------------------------------------------------------------------------------------------------------------------------#
                                                    # Grasping part
# ---------------------------------------------------------------------------------------------------------------------------- #

                        if is_grasping:

                            save_results_f(
                                         rgb_img=cam_data.get_rgb(rgb, False),
                                         depth_img=np.squeeze(cam_data.get_depth(depth)),
                                         grasp_q_img=q_img,
                                         grasp_angle_img=ang_img,
                                         no_grasps=0,
                                         grasp_width_img=width_img,
                                         point=best_grasp_point,
                                         fine_point=C_F_best_grasp_point,
                                         fine_width=F_length)

                            if campos != None:

                                if rotation != 0:
                                    F_angle = F_angle * (180 / np.pi)
                                    F_angle = F_angle - rotation
                                    F_angle = map_to_minus_90_to_90(F_angle)
                                    F_angle = F_angle * (np.pi / 180)

                                execute_grasp_6dof(
                                    arm=arm,
                                    args=args,
                                    depth_img=depth,
                                    color_intrinsics=color_intrinsics,
                                    pixel_xy=[int(data_array[0]), int(data_array[1])],
                                    depth_m=float(dis),
                                    grasp_angle_rad=F_angle,
                                    grasp_width=F_length,
                                    hand_eye_transform=hand_eye_transform,
                                )
                                wait_until_pose(arm, HOME_VIEW_POSE)
                                count = 0
                                move = 0
                                rotation = 0

                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                print("No optimal grasping point detected, move robot again")
                                aligned_depth_frame = image_bundle['aligned_depth_frame']
                                dis = depth[min_coords_squeezed[0], min_coords_squeezed[1]]
                                x, y, z = rs.rs2_deproject_pixel_to_point(intrin=color_intrinsics,
                                                                          pixel=[min_coords_squeezed[1],
                                                                                 min_coords_squeezed[0]],
                                                                          depth=dis)
                                campos = [x, y, z]

                                execute_mva_xy_move(
                                    arm=arm,
                                    hand_eye_transform=hand_eye_transform,
                                    point_cam_m=campos,
                                    args=args,
                                    label='Fallback MVA',
                                )
                                time.sleep(2)

    finally:
        None
