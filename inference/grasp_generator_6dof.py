import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp


class GraspGenerator:
    def __init__(self, saved_model_path, cam_id, visualize=False):
        self.saved_model_path = saved_model_path
        self.camera = RealSenseCamera(device_id=cam_id)
        self.depth_offset = 0.04
        self.patch_radius = 10

        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        self.cam_data = CameraData(include_depth=True, include_rgb=True)

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        self.cam_pose = np.loadtxt(self._resolve_calib_file('camera_pose.txt'), delimiter=' ')
        self.cam_depth_scale = np.loadtxt(self._resolve_calib_file('camera_depth_scale.txt'), delimiter=' ')

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        os.makedirs(homedir, exist_ok=True)
        self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")
        if not os.path.exists(self.grasp_request):
            np.save(self.grasp_request, 0)
        if not os.path.exists(self.grasp_available):
            np.save(self.grasp_available, 0)

        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    @staticmethod
    def _resolve_calib_file(filename):
        # Prefer project-local saved_data, then current working directory.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(project_root, 'saved_data', filename),
            os.path.join(os.getcwd(), 'saved_data', filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Calibration file not found: {filename}. Tried: {candidates}")

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def _predict_grasps(self, rgb, depth):
        x, _, _ = self.cam_data.get_data(rgb=rgb, depth=depth)

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(
            pred['pos'],
            pred['cos'],
            pred['sin'],
            pred['width'],
        )
        grasps = detect_grasps(q_img, ang_img, width_img)
        return grasps

    def _get_grasp_pixel(self, grasp):
        v = grasp.center[0] + self.cam_data.top_left[0]
        u = grasp.center[1] + self.cam_data.top_left[1]
        return int(v), int(u)

    def _pixel_to_camera_point(self, u, v, depth_2d):
        pos_z = depth_2d[v, u] * self.cam_depth_scale - self.depth_offset
        if pos_z <= 0:
            return None

        pos_x = (u - self.camera.intrinsics.ppx) * pos_z / self.camera.intrinsics.fx
        pos_y = (v - self.camera.intrinsics.ppy) * pos_z / self.camera.intrinsics.fy
        return np.array([pos_x, pos_y, pos_z], dtype=float)

    def _camera_to_robot_point(self, point_camera):
        rotation = self.cam_pose[0:3, 0:3]
        translation = self.cam_pose[0:3, 3]
        return rotation @ point_camera + translation

    def _collect_local_points(self, u, v, depth_2d):
        points_3d = []
        for dv in range(-self.patch_radius, self.patch_radius):
            for du in range(-self.patch_radius, self.patch_radius):
                vv = v + dv
                uu = u + du

                if vv < 0 or uu < 0 or vv >= depth_2d.shape[0] or uu >= depth_2d.shape[1]:
                    continue

                z = depth_2d[vv, uu] * self.cam_depth_scale
                if z <= 0:
                    continue

                x = (uu - self.camera.intrinsics.ppx) * z / self.camera.intrinsics.fx
                y = (vv - self.camera.intrinsics.ppy) * z / self.camera.intrinsics.fy
                points_3d.append([x, y, z])

        return np.array(points_3d, dtype=float)

    @staticmethod
    def _estimate_surface_normal(points_3d):
        if len(points_3d) < 10:
            return None

        centroid = np.mean(points_3d, axis=0)
        covariance = np.cov((points_3d - centroid).T)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            return None
        return normal / normal_norm

    @staticmethod
    def _project_to_plane(vector, normal):
        return vector - np.dot(vector, normal) * normal

    @staticmethod
    def _fallback_reference(z_axis):
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(reference, z_axis)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        return reference

    def _build_grasp_frame(self, grasp_angle, normal_camera):
        rotation = self.cam_pose[0:3, 0:3]
        normal_robot = rotation @ normal_camera
        normal_norm = np.linalg.norm(normal_robot)
        if normal_norm < 1e-8:
            return None
        normal_robot = normal_robot / normal_norm

        # Keep the tool approach direction pointing down toward the workspace.
        if normal_robot[2] > 0:
            normal_robot = -normal_robot

        z_axis = -normal_robot

        x_guess_camera = np.array([np.cos(grasp_angle), np.sin(grasp_angle), 0.0], dtype=float)
        x_guess_robot = rotation @ x_guess_camera
        x_axis = self._project_to_plane(x_guess_robot, z_axis)

        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            x_axis = self._project_to_plane(self._fallback_reference(z_axis), z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-8:
                return None

        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-8:
            return None

        y_axis = y_axis / y_norm
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

    def generate(self):
        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        depth_2d = depth[..., 0] if depth.ndim == 3 else depth
        grasps = self._predict_grasps(rgb, depth)
        if len(grasps) == 0:
            print('No grasp candidate found.')
            return False

        grasp = grasps[0]
        v, u = self._get_grasp_pixel(grasp)
        target_camera = self._pixel_to_camera_point(u, v, depth_2d)
        if target_camera is None:
            print('Invalid depth value at grasp center.')
            return False

        print('target: ', target_camera.reshape(3, 1))
        target_position = self._camera_to_robot_point(target_camera)

        points_3d = self._collect_local_points(u, v, depth_2d)
        normal_camera = self._estimate_surface_normal(points_3d)
        if normal_camera is None:
            print('Not enough points for normal estimation.')
            return False

        rpy_deg = self._build_grasp_frame(grasp.angle, normal_camera)
        if rpy_deg is None:
            print('Failed to build a valid orthonormal grasp frame.')
            return False

        # 6-DoF output for robot controller:
        # [x, y, z, roll_deg, pitch_deg, yaw_deg]
        grasp_pose = np.concatenate((target_position, rpy_deg))

        print('grasp_pose: ', grasp_pose)

        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)

        np.save(self.grasp_pose, grasp_pose)
        return True

    def run(self):
        while True:
            if np.load(self.grasp_request):
                success = self.generate()
                np.save(self.grasp_request, 0)
                np.save(self.grasp_available, 1 if success else 0)
            else:
                time.sleep(0.1)
