import chex
import jax
from jax import numpy as jnp
from typing import Sequence

# morphology imports
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification

# arena imports
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena

# environment imports
from biorobot.brittle_star.environment.light_escape.shared import BrittleStarLightEscapeEnvironmentConfiguration
from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.environment.undirected_locomotion.shared import \
    BrittleStarUndirectedLocomotionEnvironmentConfiguration


from biorobot.brittle_star.environment.undirected_locomotion.dual import BrittleStarUndirectedLocomotionEnvironment
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.environment.light_escape.dual import BrittleStarLightEscapeEnvironment
from moojoco.environment.dual import DualMuJoCoEnvironment
from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.environment.mjx_env import MJXEnv

from bsc_utils.damage import check_damage
from bsc_utils.visualization import visualize_mjcf


class EnvContainer():
    def __init__(self, config):
        self.config = config
    
    def generate_env(self):
        morph_spec, arena_conf, env_conf = _full_mjcf_configurations(config=self.config)

        self.morphology_specification = morph_spec
        self.arena_configuration = arena_conf
        self.environment_configuration = env_conf

        self.env = _create_environment(
            morphology_specification=morph_spec,
            arena_configuration=arena_conf,
            environment_configuration=env_conf,
            backend="MJX"
            )
        
    def generate_env_damaged(self):
        check_damage(arm_setup = self.config["morphology"]["arm_setup"], arm_setup_damage = self.config["damage"]["arm_setup_damage"])
        morph_spec_damage, arena_conf, env_conf = _full_mjcf_configurations(config=self.config, damage=True)

        self.morphology_specification_damage = morph_spec_damage
        self.arena_configuration = arena_conf
        self.environment_configuration = env_conf

        self.env_damage = _create_environment(
            morphology_specification=morph_spec_damage,
            arena_configuration=arena_conf,
            environment_configuration=env_conf,
            backend="MJX"
            )
    
    def clear_envs(self):
        try:
            self.env.close()
        except:
            pass
        try:
            self.env_damage.close()
        except:
            pass
        self.morphology_specification = None
        self.morphology_specification_damage = None
        self.arena_configuration = None
        self.environment_configuration = None
        self.env = None
        self.env_damage = None

    def visualize_morphology(self):
        assert self.morphology_specification, "No moprhology specification is defined yet"
        morphology = _create_morphology(morphology_specification=self.morphology_specification)
        visualize_mjcf(mjc=morphology)

    def visualize_arena(self):
        assert self.arena_configuration, "No arena configuration is defined yet"
        arena = _create_arena(arena_configuration=self.arena_configuration)
        visualize_mjcf(mjc=arena)

    def get_observation_action_space_info(self):
        # method to get e.g. dimensions from sensor and actuator space for this specific morphology
        assert self.env, "No environment is instantiated yet. Use generate_env method"
        _env_reset = jax.jit(jax.vmap(self.env.reset))
        _env_state = _env_reset(jnp.array([jax.random.PRNGKey(0)])) # PRNGKey completely unimportant, only dimensions are required
        observation_space_dim = sum([_env_state.observations[sensor].shape[-1] for sensor in self.config["environment"]["sensor_selection"]])
        actuator_space_dim = len(self.env.actuators)
        return observation_space_dim, actuator_space_dim


def _create_morphology(
        morphology_specification: BrittleStarMorphologySpecification
        ) -> MJCFBrittleStarMorphology:
    morphology = MJCFBrittleStarMorphology(
            specification=morphology_specification
            )
    return morphology

def _create_arena(
        arena_configuration: AquariumArenaConfiguration
        ) -> MJCFAquariumArena:
    arena = MJCFAquariumArena(
            configuration=arena_configuration
            )
    return arena

def _create_environment(
        morphology_specification: BrittleStarMorphologySpecification,
        arena_configuration: AquariumArenaConfiguration,
        environment_configuration: MuJoCoEnvironmentConfiguration,
        backend: str
        ) -> DualMuJoCoEnvironment:
    assert backend in ["MJC", "MJX"], "Please specify a valid backend; Either 'MJC' or 'MJX'"

    morphology = _create_morphology(
            morphology_specification=morphology_specification
            )
    arena = _create_arena(
            arena_configuration=arena_configuration
            )
    if isinstance(environment_configuration, BrittleStarUndirectedLocomotionEnvironmentConfiguration):
        env_class = BrittleStarUndirectedLocomotionEnvironment
    elif isinstance(environment_configuration, BrittleStarDirectedLocomotionEnvironmentConfiguration):
        env_class = BrittleStarDirectedLocomotionEnvironment
    else:
        env_class = BrittleStarLightEscapeEnvironment

    env = env_class.from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=environment_configuration, backend=backend
            )
    return env


def _full_mjcf_configurations(
        config: dict,
        damage: bool=False
        ):
    # specifying morphology
    if damage == True:
        check_damage(arm_setup=config["morphology"]["arm_setup"], arm_setup_damage=config["damage"]["arm_setup_damage"])

        morphology_specification = default_brittle_star_morphology_specification(
                        num_arms=len(config["damage"]["arm_setup_damage"]),
                        num_segments_per_arm=config["damage"]["arm_setup_damage"],
                        use_p_control=(config["morphology"]["joint_control"] == 'position'),
                        use_torque_control=(config["morphology"]["joint_control"] == 'torque')
                        )
    else:
        morphology_specification = default_brittle_star_morphology_specification(
                num_arms=len(config["morphology"]["arm_setup"]),
                num_segments_per_arm=config["morphology"]["arm_setup"],
                use_p_control=(config["morphology"]["joint_control"] == 'position'),
                use_torque_control=(config["morphology"]["joint_control"] == 'torque')
                )

    # specifying arena
    arena_configuration = AquariumArenaConfiguration(
            size=config["arena"]["arena_size"],
            sand_ground_color=config["arena"]["sand_ground_color"],
            attach_target=(config["environment"]["reward_type"]=="target"),
            wall_height=1.5,
            wall_thickness=0.1
            )

    # specifying environment: CHOOSE 1
    assert config["environment"]["reward_type"] in (["distance","target","light"]), "reward_type must be one of 'distance', 'target', 'light'"

    try:
        config["environment"]["render"]
    except:
        config["environment"]["render"] = {"render_size": [ 480, 640 ], "camera_ids": [ 0, 1 ]}

    if config["environment"]["reward_type"] == "distance":
        environment_configuration = BrittleStarUndirectedLocomotionEnvironmentConfiguration(
            # If this value is > 0 then we will add randomly sampled noise to the initial joint positions and velocities 
            joint_randomization_noise_scale=config["environment"]["joint_randomization_noise_scale"],
            render_mode="rgb_array",  # Visualization mode 
            simulation_time=config["environment"]["simulation_time"],  # Number of seconds per episode 
            num_physics_steps_per_control_step=config["environment"]["num_physics_steps_per_control_step"],  # Number of physics substeps to do per control step 
            time_scale=2,    # Integer factor by which to multiply the original physics timestep of 0.002,
            camera_ids=config["environment"]["render"]["camera_ids"],   # Which camera's to render (all the brittle star environments contain 2 cameras: 1 top-down camera and one close-up camera that follows the brittle star),
            render_size=config["environment"]["render"]["render_size"],  # Resolution to render with ((height, width) in pixels)
            color_contacts=config["environment"]["color_contacts"]
            )

    elif config["environment"]["reward_type"] == "target":
        environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
            # Distance to put our target at (targets are spawned on a circle around the starting location with this given radius).
            target_distance=config["environment"]["target_distance"],
            joint_randomization_noise_scale=config["environment"]["joint_randomization_noise_scale"],
            render_mode="rgb_array",
            simulation_time=config["environment"]["simulation_time"],
            num_physics_steps_per_control_step=config["environment"]["num_physics_steps_per_control_step"],
            time_scale=2,
            camera_ids=config["environment"]["render"]["camera_ids"],
            render_size=config["environment"]["render"]["render_size"],
            color_contacts=config["environment"]["color_contacts"]
            )

    elif config["environment"]["reward_type"] == "light":
        environment_configuration = BrittleStarLightEscapeEnvironmentConfiguration(
            joint_randomization_noise_scale=config["environment"]["joint_randomization_noise_scale"],
            light_perlin_noise_scale=config["environment"]["light_perlin_noise_scale"],
            # If this value is > 0, we will add perlin noise to the generated light map. Otherwise, the light map is a simple linear gradient.
            # Please only provide integer factors of 200.
            render_mode="rgb_array",
            simulation_time=config["environment"]["simulation_time"],
            num_physics_steps_per_control_step=config["environment"]["num_physics_steps_per_control_step"],
            time_scale=2,
            camera_ids=config["environment"]["render"]["camera_ids"],
            render_size=config["environment"]["render"]["render_size"],
            color_contacts=config["environment"]["color_contacts"]
            )
    
    return morphology_specification, arena_configuration, environment_configuration



