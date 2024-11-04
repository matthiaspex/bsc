# Visualisation module
import numpy as np
import mediapy as media
from typing import List
from moojoco.environment.base import MuJoCoEnvironmentConfiguration, BaseEnvState
from moojoco.mjcf.component import MJCFRootComponent
from moojoco.environment.mjx_env import MJXEnvState
import cv2
import mujoco
import logging
import matplotlib.pyplot as plt
import imageio
import copy
from jax import numpy as jnp


def visualize_mjcf(
        mjcf: MJCFRootComponent
        ) -> None:
    model = mujoco.MjModel.from_xml_string(mjcf.get_mjcf_str())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    media.show_image(renderer.render())


def post_render(
        render_output: List[np.ndarray],
        environment_configuration: MuJoCoEnvironmentConfiguration
        ) -> np.ndarray:
    if render_output is None:
        # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
        return None

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [np.concatenate(env_frames, axis=1) for env_frames in frames_per_env]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return render_output[:, :, ::-1]  # RGB to BGR

def show_video(
        images: List[np.ndarray | None],
        fps: int = 60
        ) -> str | None:
    # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
    filtered_images = [image for image in images if image is not None]
    num_nones = len(images) - len(filtered_images)
    if num_nones > 0:
        logging.warning(f"env.render produced {num_nones} None's. Resulting video might be a bit choppy (consquence of https://github.com/google-deepmind/mujoco/issues/1379).")
    return media.show_video(images=filtered_images, fps=fps)


# Making videos to upload to wandb afterwards
def create_video(
        frames: List[np.ndarray],
        framerate: float,
        out_path: str
        ) -> None:
    height, width, _ = frames[0].shape
    size = (width, height)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), framerate, size) # fourcc was at first *'mp4v'
    # fourcc h264 and avc1 will have better odds of actually visualising in chrome browser, but not by default supported by opencv
    for frame in frames:
        writer.write(frame)
    writer.release()


def generate_timestep_joint_angle_plot_data(
        arms,
        vectorized_env_state: BaseEnvState
):
    joint_angles_ip_t = []
    joint_angles_oop_t = []
    j = 0

    for n in arms:
        if n != 0:
            joint_angles_ip_t.append(vectorized_env_state.observations["joint_position"][0][j*2*n:(j+1)*2*n:2])
            joint_angles_oop_t.append(vectorized_env_state.observations["joint_position"][0][j*2*n+1:(j+1)*2*n+1:2])
            j += 1
    return joint_angles_ip_t, joint_angles_oop_t


def plot_ip_oop_joint_angles(
        joint_angles_ip: List,
        joint_angles_oop: List,
        show_plot: bool=False
):
    joint_angles_ip = np.array(joint_angles_ip)
    joint_angles_oop = np.array(joint_angles_oop)
    # print(f"shape explanation: (t, number of arms, number of segment) = {joint_angles_ip.shape}")
    t, num_arms, num_segments = joint_angles_ip.shape

    plt.rcParams['figure.figsize'] = (5*num_arms,5*num_segments)

    fig, axes = plt.subplots(num_segments,num_arms, squeeze = False)
    # squeeze allows damaged morphologies with only 1 arm to also give 2D axes object

    for i in range(num_segments):
        for j in range(num_arms):
            axes[i][j].plot(joint_angles_ip[:,j,i], joint_angles_oop[:,j,i])
            axes[i][j].set_title(f"Joint angles for segment {i} in arm {j}")
            axes[i][j].set_xlabel("IP joint angle [rad]")
            axes[i][j].set_ylabel("OOP joint angle [rad]")
    if show_plot:
        plt.show()
    return fig, axes

def save_video_from_raw_frames(
        frames,
        fps: int,
        file_path: str
):
    imgio_kargs = {
        'fps': fps, 'quality': 10, 'macro_block_size': None, 'codec': 'h264',
        'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
        }
    writer = imageio.get_writer(file_path, **imgio_kargs)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def move_camera(state: MJXEnvState) -> MJXEnvState:
    # Move the top down camera down
    new_mjx_model = state.mjx_model.replace(
            cam_pos=state.mjx_model.cam_pos.at[0, 2].set(6)
            )
    return state.replace(mjx_model=new_mjx_model)


def change_alpha(
        state: MJXEnvState,
        brittle_star_alpha: float,
        background_alpha: float
        ) -> MJXEnvState:
    """
    Used to vary opacity of background and brittle star.
    """
    brittle_star_geom_ids = jnp.array(
            [geom_id for geom_id in range(state.mj_model.ngeom) if
             "BrittleStarMorphology" in state.mj_model.geom(geom_id).name]
            )
    background_geom_ids = jnp.array(
            [geom_id for geom_id in range(state.mj_model.ngeom) if
             "BrittleStarMorphology" not in state.mj_model.geom(geom_id).name]
            )

    geom_rgba = copy.deepcopy(state.mjx_model.geom_rgba)
    geom_rgba[brittle_star_geom_ids, 3] = brittle_star_alpha
    geom_rgba[background_geom_ids, 3] = background_alpha
    new_mjx_model = state.mjx_model.replace(geom_rgba=geom_rgba)
    # noinspection PyUnresolvedReferences
    state = state.replace(mjx_model=new_mjx_model)
    return state




def create_histogram(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency', color='blue', edgecolor='black', alpha=0.7):
    """
    Creates a histogram plot and returns the figure object.

    Parameters:
    - data: array-like, the data to plot
    - bins: int, number of bins in the histogram
    - color: str, color of the bars
    - edgecolor: str, color of the bar edges
    - alpha: float, transparency of the bars
    - title: str, title of the plot
    - xlabel: str, label for the x-axis
    - ylabel: str, label for the y-axis

    Returns:
    - fig: matplotlib.figure.Figure, the figure object of the histogram
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


        