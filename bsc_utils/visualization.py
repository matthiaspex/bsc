# Visualisation module
import biorobot
import numpy as np
import mediapy as media
from typing import Any, Callable, Sequence, Union, List
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.mjcf.component import MJCFRootComponent
import cv2
import mujoco
import logging
import matplotlib.pyplot as plt
import imageio
from PIL import Image


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



def plot_ip_oop_joint_angles(
        joint_angles_ip: List,
        joint_angles_oop: List
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
            axes[i][j].set_xlabel("In plane joint angle [rad]")
            axes[i][j].set_ylabel("Out of plane joint angle [rad]")
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


def save_image_from_raw_frames(
        frames,
        number_of_frames: int,
        file_path: str = None,
        show_image: bool = False
) -> Image:
    frames_sel = frames[::len(frames)//number_of_frames]
    img = Image.fromarray(frames_sel[0], 'RGB')
    for frame in frames_sel[1:]:
        img_add = Image.fromarray(frame, 'RGB')
        img = Image.blend(img, img_add, 0.5)

    if file_path:
        img.save(file_path)
    if show_image:
        img.show()
        