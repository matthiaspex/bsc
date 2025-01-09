from ast import List
import os
import subprocess
import logging
from typing import Sequence, Optional, Tuple
import yaml
from collections import Counter

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
        parallel_constant: bool=False,
        force_single_direction: Optional[Tuple[bool, str, int]] = None
)->Sequence:
    """
    Returns array with cartesian positions of targets on a circle with a certain radius
    for rowing and reverse rowing positions based on starting position.
    parallel_dim: can provide popsizes for parallelized environment resets.
    parallel_constant: same target positions for all parallel environments or not.

    inputs:
    - force_single_direction: default None, otherwise Tuple (True/False, "rowing"/"reverse_rowing", arm_index)
    
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
    
    if force_single_direction != None and force_single_direction[0] == True:
        if force_single_direction[1] == "rowing":
            angle = jnp.arange(0,359, 360//5)[force_single_direction[2]]
        elif force_single_direction[1] == "reverse_rowing":
            angle = jnp.arange(360//10,359,360//5)[force_single_direction[2]]
        else:
            raise ValueError("string at position 1 in tuple force_single_direction in config file should be 'rowing' or 'reverse_rowing'")
        angles = jnp.ones((parallel_dim,1)) * angle # in case of force_single_direction: overwrite the angles already generated.


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
        if not "unit_xy_direction_to_target" in sensor_selection_new:
            sensor_selection_new.append("unit_xy_direction_to_target")
        if not "xy_distance_to_target" in sensor_selection_new:
            sensor_selection_new.append("xy_distance_to_target")
        if not "disk_rotation" in sensor_selection_new:
            sensor_selection_new.append("disk_rotation")
    if config["environment"]["reward_type"] == "light":
        if not "segment_light_intake" in sensor_selection_new:
            sensor_selection_new.append("segment_light_intake")

    sensor_selection_new = tuple(sensor_selection_new)
    config["environment"]["sensor_selection"] = sensor_selection_new
    
    return config


def order_sensor_selection_in_config(config:dict) -> dict:
    """
    Will arange the sensor selection into the following order:
    possible_sensors = ['joint_position', 'joint_velocity', 'joint_actuator_force', 'actuator_force',
     'disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity', 'segment_contact',
     'unit_xy_direction_to_target', 'xy_distance_to_target',
     'segment_light_intake'
    ]
    """

    possible_sensors = ['joint_position', 'joint_velocity', 'joint_actuator_force', 'actuator_force',\
     'disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity', 'segment_contact',\
     'unit_xy_direction_to_target', 'xy_distance_to_target',\
     'segment_light_intake'\
    ]

    sensor_selection_ordered = [sensor for sensor in possible_sensors if sensor in config["environment"]["sensor_selection"]]
    sensor_selection_ordered = tuple(sensor_selection_ordered)
    config["environment"]["sensor_selection"] = sensor_selection_ordered
    
    return config



def complete_config_with_defaults(config):
    """
    For visualising older experiments with incomplete config files: complete them with default values
    """
    try:
        config["evolution"]["penal_expr"]
    except:
        config["evolution"]["penal_expr"] = "nopenal"

    try:
        config["evolution"]["efficiency_expr"]
    except:
        config["evolution"]["efficiency_expr"] = config["evolution"]["fitness_expr"]

    try:
        config["controller"]["hebbian"]
    except:
        config["controller"]["hebbian"] = False

    try:
        config["controller"]["num_hebbian_steps_per_control_step"]
    except:
        config["controller"]["num_hebbian_steps_per_control_step"] = 1

    try:
        config["evolution"]["centered_rank"]
    except:
        config["evolution"]["centered_rank"] = True

    try:
        config["training"]
    except:
        config["training"] = {}
    
    try:
        config["training"]["target"]
    except:
        config["training"]["target"] = {}
    
    try:
        config["training"]["target"]["force_single_direction"]
    except:
        config["training"]["target"]["force_single_direction"] = [False, "rowing", 0]

    try:
        config["controller"]["decentralized"]
    except:
        config["controller"]["decentralized"] = {}
    
    try:
        config["controller"]["decentralized"]["decentralized_on"]
    except:
        config["controller"]["decentralized"]["decentralized_on"] = False

    try:
        config["morphology"]["replace_joint_stiffness"]
    except:
        config["morphology"]["replace_joint_stiffness"] = (False, 0.1)
    
    try:
        config["morphology"]["replace_joint_damping"]
    except:
        config["morphology"]["replace_joint_damping"] = (False, 0.5)
    
    try:
        config["morphology"]["replace_joint_armature"]
    except:
        config["morphology"]["replace_joint_armature"] = (False, 0.02)

    try:
        config["controller"]["biases"]
    except:
        config["controller"]["biases"] = True
    
    try:
        config["controller"]["bias_decay"]
    except:
        config["controller"]["bias_decay"] = 1.0

    try:
        config["controller"]["kernel_decay"]
    except:
        config["controller"]["kernel_decay"] = 1.0
    
    try:
        config["controller"]["kernel_clipping"]
    except:
        config["controller"]["kernel_clipping"] = False

    try:
        config["controller"]["multiplicative_plasticity"]
    except:
        config["controller"]["multiplicative_plasticity"] = False

    try:
        config["controller"]["presynaptic_competition"]
    except:
        config["controller"]["presynaptic_competition"] = False

    try:
        config["controller"]["anti_zero_crossing"]
    except:
        config["controller"]["anti_zero_crossing"] = False

    return config



def check_sensor_selection_order(
        sensor_selection
):
    """
    For the brittle star environment, the sensors are ordered in a very specific way.
    This function checks wether the sensor selection is composed correctly.
    The order of the sensors should be
    ['joint_position', 'joint_velocity', 'joint_actuator_force', 'actuator_force',
     'disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity', 'segment_contact',
     'unit_xy_direction_to_target', 'xy_distance_to_target',
     'segment_light_intake'
    ]

    Checks if all elements of `sensor_selection` appear in the above list in the same order.

    Returns nothing if sensor_selection order is correct
    Raises error otherwise.
    """
    possible_sensors = ['joint_position', 'joint_velocity', 'joint_actuator_force', 'actuator_force',\
     'disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity', 'segment_contact',\
     'unit_xy_direction_to_target', 'xy_distance_to_target',\
     'segment_light_intake'\
    ]

    # check wether all provided sensors are valid sensors
    for sensor in sensor_selection:
        if not sensor in possible_sensors:
            raise ValueError(f"The provided sensor '{sensor}' is not a valid sensor for the agent.\n\
    Check the valid brittle star sensor by calling the help() function")
        
    # # check whether none of the sensors appear twice
    # counts = Counter(sensor_selection)
    # if any(count > 1 for count in counts.values()):
    #     raise ValueError("Some of the sensors in sensor selection appear twice")



    # check whether sensors are in the correct order
    it = iter(possible_sensors)  # Create an iterator for the longer list
    if all(element in it for element in sensor_selection):
        pass
    else:
        raise ValueError("The provided sensors are not in the correct order.\n\
    Check the correct order by calling the help() function")



def calculate_arm_target_allignment_factors(
        unit_xy_direction_to_target: chex.Array,
        num_arms = 5
) -> chex.Array:
    """
    Inputs
    - unit_xy_direction_to_target: max 1 per simulation environment, can be 1D with dim (2,) or 2D with dim (#popsize, 2)
    - num_arms: usually 5 arms are present
    Outputs
    - returns an array with 5 values between -1 and 1
        -> 1: arm pointed towards the target
        -> 0: arm direction perpendicular to target direction
        -> -1: arm direction opposite to target
    if unit_xy_direction_to_target had a popsize dim on axis = 0, the output will also have a popsize dim on axis = 0
    """
    arm_dir = []
    angle = 2*jnp.pi/num_arms

    for i in range(num_arms):
        arm_dir.append(jnp.array([jnp.cos(i*angle), jnp.sin(i*angle)]))

    arm_dir = jnp.array(arm_dir)


    vector = False
    # If there are multiple parallel dimensions, copy the arm directions to all these parallel dimensions.
    if len(unit_xy_direction_to_target.shape) == 2:
        vector = True
        arm_dir_expanded = jnp.expand_dims(arm_dir, axis = 0)
        arm_dir_expanded = jnp.tile(arm_dir_expanded, (unit_xy_direction_to_target.shape[0],1,1))


    vectorized_dot = jax.vmap(jnp.dot)

    if vector:
        arm_projections = jnp.zeros((unit_xy_direction_to_target.shape[0], 5))
    else:
        arm_projections = jnp.zeros(5)

    for i in range(num_arms):
        if vector:
            dot = vectorized_dot(arm_dir_expanded[:,i,:], unit_xy_direction_to_target)
            arm_projections = arm_projections.at[:,i].set(dot)
        else:
            dot = jnp.dot(arm_dir[i,:], unit_xy_direction_to_target)
            arm_projections = arm_projections.at[i].set(dot)


    return arm_projections


def decay_kernel_bias_dict(
        param_dict: dict,
        kernel_decay: float=1.0,
        bias_decay: float=1.0
    ) -> dict:
    """
    Takes policy params or arm states as input and all the leaves which have "kernel" and/or "bias"
    in their path will be decayed with the provided factor

    inputs:
    - param_dict: the dictionary with policy params or arm states
    - kernel_decay: the factor to apply to the kernel_decay; if 1: no decay applied
    - bias_decay: the factor to apply to the bias_decay; if 1: no decay applied

    output:
    - return dict (pytree) with modified weights and/or biases
    """
    def modify_leaves_with_kernel_in_path(path, leaf):
        if "kernel" in jax.tree_util.keystr(path):
            leaf = leaf*kernel_decay
            return leaf  # Modify leaf
        return leaf  # Keep unchanged otherwise

    def modify_leaves_with_bias_in_path(path, leaf):
        if "bias" in jax.tree_util.keystr(path):
            leaf = leaf*bias_decay
            return leaf  # Modify leaf
        return leaf  # Keep unchanged otherwise
    
    if kernel_decay != 1.0:
        param_dict = jax.tree_util.tree_map_with_path(modify_leaves_with_kernel_in_path, param_dict)
    if bias_decay != 1.0:
        param_dict = jax.tree_util.tree_map_with_path(modify_leaves_with_bias_in_path, param_dict)

    return param_dict


def clip_kernel_biases_dict(
        param_dict: dict,
        kernel_min: Optional[float]=None,
        kernel_max: Optional[float]=None,
        bias_min: Optional[float]=None,
        bias_max: Optional[float]=None
) -> dict:
    """
    Takes policy params or arm states as input and all the leaves which have "kernel" and/or "bias" in their path
    will be clipped to not go outside the provided interval (or only max/min value)
    
    inputs:
    - param_dict: the dictionary with policy params or arm states
    - kernel_min: the lowest value in the kernel arrays which will still occur in the resulting dict
    - kernel_max: the highest value in the kernel arrays which will still occur in the resulting dict
    - bias_min: the lowest value in the bias arrays which will still occur in the resulting dict
    - bias_max: the highest value in the bias arrays which will still occur in the resulting dict
    
    output:
    - return dict (pytree) with the modified weights and/or biases
    """
    def clip_leaves_with_kernel_in_path(path, leaf):
        if "kernel" in jax.tree_util.keystr(path):
            leaf = jnp.clip(leaf, min=kernel_min, max=kernel_max)
            return leaf  # Modify leaf
        return leaf  # Keep unchanged otherwise

    def clip_leaves_with_bias_in_path(path, leaf):
        if "bias" in jax.tree_util.keystr(path):
            leaf = jnp.clip(leaf, min=bias_min, max=bias_max)
            return leaf  # Modify leaf
        return leaf  # Keep unchanged otherwise
    
    if kernel_min != None or kernel_max != None:
        if kernel_min and kernel_max:
            assert kernel_min < kernel_max, "kernel_min should be smaller than kernel_max"
        param_dict = jax.tree_util.tree_map_with_path(clip_leaves_with_kernel_in_path, param_dict)

    if bias_min != None or bias_max != None:
        if bias_min and bias_max:
            assert bias_min < bias_max, "bias_min should be smaller than bias_max"
        param_dict = jax.tree_util.tree_map_with_path(clip_leaves_with_bias_in_path, param_dict)

    return param_dict


def presynaptic_competition_rescale(
        param_dict: dict,

):
    """
    Based on Fung and Fukai (2023) eq (3)
    For every output node (column in the kernel matrix), check whether any weight surpasses value 1.0
    If it surpasses, than divide entire column by the maximum value.

    Input:
    - param_dict: a pytree containing the weight kernels (can be arm_states object as well)

    Output:
    - dict (pytree) containing the rescaled weights
    """
    def rescale_kernel_columns(path, leaf):
        if "kernel" in jax.tree_util.keystr(path):
            num_rows = leaf.shape[0]
            maximums = jnp.max(jnp.abs(leaf), axis=0) # also look at biggest negative values. Also taken into account for the rescaling
            maximums = jnp.where(maximums < 1., 1., maximums) # if maximum in column is less then 1., division by 1.
            # will occur in that column. If it is greater then 1, division by the maximum will occur in that column.
            maximums_2D = jnp.tile(maximums, (num_rows,1))
            return leaf/maximums_2D  # Modify leaf
            # division is automatically by positive value, so no signs will switch (since maximums_2D only comes from jnp.abs(leaf) values)
        return leaf  # Keep unchanged otherwise

    param_dict = jax.tree_util.tree_map_with_path(rescale_kernel_columns, param_dict)
    
    return param_dict


def multimodal_normal_sampling(
        rng: chex.PRNGKey,
        means: List,
        stds: List,
        sample_sizes: List,
        trunc_mins: List,
        trunc_maxs: List
) -> chex.Array:
    """
    Samples from a multimodal normal distribution.
    Inputs:
    - rng: jax random number generator key
    - means: means of the different normal distributions
    - stds: standard deviations of the different normal distributions
    - sample_sizes: how many samples to pick randomly from every distribution
    - trunc_min: to prevent large initialisations, truncate the distributions minimums
    - trunc_max: to prevent large initialisations, truncate the distribution maximums
    The lengths of all inputs should be identical
    """
    lengths = []
    lengths.append(len(means))
    lengths.append(len(stds))
    lengths.append(len(sample_sizes))
    lengths.append(len(trunc_mins))
    lengths.append(len(trunc_maxs))
    num = lengths[0]

    assert all(element == lengths[0] for element in lengths), "make sure all the provided lists have the same length"

    rng, rng_tmp = jax.random.split(rng, 2)
    rngs = jax.random.split(rng_tmp, num)

    x_list = []
    for i in range(num):
        x_tmp = jax.random.truncated_normal(rngs[i], (trunc_mins[i]-means[i])/stds[i], (trunc_maxs[i]-means[i])/stds[i], sample_sizes[i]) * stds[i] + means[i]
        x_list.append(x_tmp)

    x = jnp.concatenate(x_list)

    rng, rng_permutation = jax.random.split(rng, 2)
    x = jax.random.permutation(rng_permutation, x)

    return x




def prune_pytree(pytree, obligatory_keyword="kernel"):
    """
    Keeps only the leaves from a PyTree where the path contains `obligatory_keyword`.

    Args:
        pytree: The input PyTree (nested dictionary, tuple, etc.).
        obligatory_keyword: String that must be in the path for the leaf to be kept.

    Returns:
        A pruned PyTree containing only leaves with paths that include the `obligatory_keyword`.
    """
    def filter_fn(path, leaf):
        """Return True if the obligatory keyword is in the path."""
        return obligatory_keyword in str(path)

    # Use `tree_map_with_path` to retain only elements that contain the `obligatory_keyword`
    pruned_pytree = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf if filter_fn(path, leaf) else None,
        pytree
    )

    # Recursively prune any `None` values from the structure
    def remove_nones(x):
        if isinstance(x, dict):
            return {k: remove_nones(v) for k, v in x.items() if v is not None}
        elif isinstance(x, tuple):
            return tuple(remove_nones(v) for v in x if v is not None)
        else:
            return x

    # Apply the pruning
    return remove_nones(pruned_pytree)
   

