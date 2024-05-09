import os
import subprocess
import logging
from typing import Sequence
import yaml

import jax
from jax import numpy as jnp
import chex

def check_GPU_access(
        interface = None
        ):
    try:
        if subprocess.run('nvidia-smi').returncode:
            raise RuntimeError(
                    'Cannot communicate with GPU. '
                    'Make sure you are using a GPU Colab runtime. '
                    'Go to the Runtime menu and select Choose runtime type.'
                    )

        # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
        # This is usually installed as part of an Nvidia driver package, but the Colab
        # kernel doesn't install its driver via APT, and as a result the ICD is missing.
        # (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
        NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
        if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
            with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
                f.write(
                        """{
                                "file_format_version" : "1.0.0",
                                "ICD" : {
                                    "library_path" : "libEGL_nvidia.so.0"
                                }
                            }
                            """
                        )

        # Configure MuJoCo to use the EGL rendering backend (requires GPU)
        print('Setting environment variable to use GPU rendering:')
        if interface != 'HPC': # otherwise these are defined in batch script used in hpc
            os.environ["MUJOCO_GL"] = "egl"
            xla_flags = os.environ.get('XLA_FLAGS', '')
            xla_flags += ' --xla_gpu_triton_gemm_any=True'
            os.environ['XLA_FLAGS'] = xla_flags

        # Check if jax finds the GPU
        import jax

        print(jax.devices('gpu'))
    except Exception:
        logging.warning("Failed to initialize GPU. Everything will run on the cpu.")

    try:
        print('Checking that the mujoco installation succeeded:')
        import mujoco

        mujoco.MjModel.from_xml_string('<mujoco/>')
    except Exception as e:
        raise e from RuntimeError(
                'Something went wrong during installation. Check the shell output above '
                'for more information.\n'
                'If using a hosted Colab runtime, make sure you enable GPU acceleration '
                'by going to the Runtime menu and selecting "Choose runtime type".'
                )

    print('MuJoCo installation successful.')


def load_config_from_yaml(
        yaml_path: str
        ):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def store_config_and_policy_params(
        file_name:str,
        cfg: dict,
        policy_params
        ):
    with open(file_name + ".yaml", "w") as f:
        dump = yaml.dump(cfg)
        f.write(dump)
    jnp.save(file_name + ".npy", policy_params)


def get_run_name_from_config(
        cfg: dict
):
    """
    If an f-string like format is provided in the config file, which takes other config information:
    e.g.    run_name_format: "{reward_type} arms {arm_setup} popsize {es_popsize} {notes}" 
    This string can be read out using this function.
    Only information up untill 3 indents in the config file can be read out this way.
    """
    config_flat = {}
    for k, v in cfg.items():
        if not isinstance(v, dict):
            config_flat[k] = v
        else:
            for l, w in v.items():
                if not isinstance(w, dict):
                    config_flat[l] = w
                else:
                    for m, u in w.items():
                        if not isinstance(u, dict):
                            config_flat[m] = u

    run_name = cfg["experiment"]["run_name_format"].format(**config_flat)
    return run_name



def get_target_positions(
        rng: chex.PRNGKey,
        distance: float,
        num_rowing: int,
        num_reverse_rowing: int,
        num_random_positions: int=0,
        parallel_dim: int=1,
        parallel_constant: bool=False
)->Sequence:
    """
    Returns array with cartesian positions of targets on a circle with a certain radius
    for rowing and reverse rowing positions based on starting position.
    parallel_dim: can provide popsizes for parallelized environment resets.
    parallel_constant: same target positions for all parallel environments or not.
    --------------
    returns: list[array(parallel_dim, 3)]
    --> iterate over list: each list element is a parallelised vector of 3D-target positions which can be vmapped
    --> in case parallel_constant = True: everything should be constant along parallel_dim dimension.
    """
    # angles target location
    rng, rng_rowing, rng_reverse_rowing = jax.random.split(rng, 3)
    vectorized_permutation = jax.vmap(jax.random.permutation)

    rowing_angles = jnp.arange(0,359, 360//5)
    rowing_angles = jnp.expand_dims(rowing_angles, axis = 0)
    rowing_angles = jnp.repeat(rowing_angles, parallel_dim, axis = 0)
    if parallel_constant == False:
        rng_rowing_vect = jax.random.split(rng_rowing, (parallel_dim,))
        rowing_angles = vectorized_permutation(rng_rowing_vect, rowing_angles)

    reverse_rowing_angles = jnp.arange(360//10,359,360//5)
    reverse_rowing_angles = jnp.expand_dims(reverse_rowing_angles, axis = 0) 
    reverse_rowing_angles = jnp.repeat(reverse_rowing_angles, parallel_dim, axis = 0)
    if parallel_constant == False:
        rng_reverse_rowing_vect = jax.random.split(rng_reverse_rowing, (parallel_dim,))
        reverse_rowing_angles = vectorized_permutation(rng_reverse_rowing_vect, reverse_rowing_angles)

    rng, rng_rowing, rng_reverse_rowing, rng_random = jax.random.split(rng, 4)
    rowing_angles_sel = jax.random.choice(rng_rowing, rowing_angles, shape=(num_rowing,), replace=False, axis = 1)
    reverse_rowing_angles_sel = jax.random.choice(rng_reverse_rowing, reverse_rowing_angles, shape=(num_reverse_rowing,), replace=False, axis = 1)
    if parallel_constant == False:
        random_angles = jax.random.uniform(rng_random, shape=(parallel_dim, num_random_positions), minval=0, maxval=360)
    elif parallel_constant == True:
        random_angles = jax.random.uniform(rng_random, shape=(num_random_positions,), minval=0, maxval=360)
        random_angles = jnp.expand_dims(random_angles, axis = 0)
        random_angles = jnp.repeat(random_angles, parallel_dim, axis = 0)

    if num_reverse_rowing == 0 and num_rowing == 0:
        angles = random_angles
    elif num_rowing == 0:
        angles = jnp.concatenate([reverse_rowing_angles_sel, random_angles], axis = -1)
    elif num_reverse_rowing == 0:
        angles = jnp.concatenate([rowing_angles_sel, random_angles], axis = -1)
    else:
        angles = jnp.concatenate([rowing_angles_sel, reverse_rowing_angles_sel, random_angles], axis = -1)
    angles = jnp.radians(angles)

    target_positions = []
    for j in range(angles.shape[-1]):
        x = distance*jnp.cos(angles[:,j])
        y = distance*jnp.sin(angles[:,j])
        z = jnp.zeros(parallel_dim)
        positions = jnp.array([x,y,z]).T # this way, the parallelized dimension to vmap is the first dim
        target_positions.append(positions)


    return target_positions 



def complete_sensor_selection(config):
    """
    In config files, the target or light information does not need to be specified in the sensor_selection tuple.
    You can call this and when the reward_type is target or light, the appropriate observations will be added to the list.
    """
    # Complete sensor selection with information about target location and light sensing
    sensor_selection_new = list(config["environment"]["sensor_selection"])
    if config["environment"]["reward_type"] == "target":
        sensor_selection_new.append("unit_xy_direction_to_target")
        sensor_selection_new.append("xy_distance_to_target")
        sensor_selection_new.append("disk_rotation")
    if config["environment"]["reward_type"] == "light":
        sensor_selection_new.append("segment_light_intake")


    
    sensor_selection_new = tuple(sensor_selection_new)
    return sensor_selection_new