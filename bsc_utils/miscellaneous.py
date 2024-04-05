import os
import subprocess
import logging
import yaml
from jax import numpy as jnp

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