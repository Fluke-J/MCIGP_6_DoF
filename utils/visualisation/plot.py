import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps

warnings.filterwarnings("ignore")

# 6-DoF visualization note:
# The current plotting utilities are 2D only. They display RGB/depth images,
# grasp rectangles, and quality maps from planar grasp detection.
#
# When the 6-DoF pipeline is wired in, this file is a good place to add a
# dedicated 3D visualization function, for example `plot_grasp_3d(...)`.
# That future function should receive precomputed 3D data from
# `inference/grasp_generator_6dof.py` rather than recomputing it here.
#
# Suggested future inputs:
# - points_3d: local point cloud around the selected grasp
# - target_position: grasp center in robot or camera coordinates
# - normal_camera / normal_robot: estimated surface normal
# - rpy_deg: grasp orientation as roll, pitch, yaw
# - grasp_frame: optional 3D axes for drawing the grasp pose
#
# Suggested future outputs:
# - a Matplotlib 3D subplot or Open3D visualization
# - saved debug image for the chosen 6-DoF grasp pose
# - optional side-by-side 2D + 3D debug view


def plot_results(
        fig,
        rgb_img,
        rgb_img_new=None,
        rgb_img_og=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        depth_img=None,
        no_grasps=1,
        grasp_width_img= None,
        point = None,
        fine_point = None,
        fine_width = None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point, fine_point = fine_point, fine_width=fine_width)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img_new)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
            ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=3.5)
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(rgb_img_og)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=3)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')

    plt.pause(0.1)
    fig.canvas.draw()


def plot_grasp(
        fig,
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()

    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    plt.pause(0.1)
    fig.canvas.draw()

    if save:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.savefig('results/{}.png'.format(time))


def plot_grasp_3d(
        fig=None,
        points_3d=None,
        target_position=None,
        normal=None,
        rpy_deg=None,
        grasp_frame=None,
        save=False,
        save_path='results/grasp_3d.png'
):
    """
    Placeholder for future 6-DoF grasp visualization.

    Planned responsibility:
    - show the local 3D point cloud near the selected grasp
    - draw the grasp center and surface normal
    - draw the grasp frame/orientation from roll, pitch, yaw
    - optionally save the 3D debug view for analysis

    Intended caller:
    - inference/grasp_generator_6dof.py after it computes the final 6-DoF pose

    Current behavior:
    - intentionally does nothing so existing 2D workflows remain unchanged
    """
    # TODO: implement 3D visualization once the 6-DoF data flow is finalized.
    # A likely first version would use `fig.add_subplot(111, projection='3d')`
    # and draw:
    # - `points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]`
    # - a marker at `target_position`
    # - a quiver for `normal`
    # - three colored quivers for `grasp_frame`
    #
    # Keeping this as a stub for now makes the intended integration point clear
    # without changing current behavior.
    return None


def save_results(
            rgb_img,
            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)

    ax.axis('off')
    fig.savefig('results/rgb.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=1)

    ax.axis('off')
    fig.savefig('results/grasp.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=300)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    for g in gs:
        # g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=1)

    ax.axis('off')
    # plt.colorbar(plot, ax=ax, shrink=0.796)
    fig.savefig('results/quality.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=300)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hot', vmin=-np.pi / 2, vmax=np.pi / 2)

    ax.axis('off')
    # plt.colorbar(plot, ax=ax, shrink=0.796)
    fig.savefig('results/angle.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='hot', vmin=0, vmax=100)
    ax.axis('off')
    # plt.colorbar(plot, ax=ax, shrink=0.796)
    fig.savefig('results/width.png', bbox_inches='tight', pad_inches=0)

    fig.canvas.draw()
    plt.close(fig)


def save_results_s(
            rgb_img,
            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=15)

    ax.axis('off')
    fig.savefig('results/grasp_s.png',bbox_inches='tight', pad_inches=0)
    fig.canvas.draw()
    plt.close(fig)

def save_results_o(
            rgb_img,
            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=15)

    ax.axis('off')
    fig.savefig('results/grasp_o.png',bbox_inches='tight', pad_inches=0)


    fig.canvas.draw()
    plt.close(fig)

def save_results_f(
            rgb_img,
            rgb_img_new,
            rgb_img_og,
            grasp_q_img,
            grasp_angle_img,
            depth_img=None,
            no_grasps=1,
            grasp_width_img=None,
            point=None,
            fine_point=None,
            fine_width=None
    ):
    # Fallbacks if not provided
    if rgb_img_new is None:
        rgb_img_new = rgb_img

    if rgb_img_og is None:
        rgb_img_og = rgb_img

    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, point=point,
                       fine_point=fine_point, fine_width=fine_width)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img_new)

    ax.axis('off')
    fig.savefig('results/rgb.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10),dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img_og)

    ax.axis('off')
    fig.savefig('results/rgb_og.png',bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(10, 10), dpi=600)
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img_og)
    for g in gs:
        g.plot(ax)
        ax.plot(g.center[1], g.center[0], 'o', color='orange', markersize=10)

    ax.axis('off')
    fig.savefig('results/grasp_f.png',bbox_inches='tight', pad_inches=0)
    fig.canvas.draw()
    plt.close(fig)
