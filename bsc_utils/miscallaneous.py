import os
import subprocess
import logging

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