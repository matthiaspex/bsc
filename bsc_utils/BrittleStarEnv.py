import gymnasium
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



def create_morphology(
        morphology_specification: BrittleStarMorphologySpecification
        ) -> MJCFBrittleStarMorphology:
    morphology = MJCFBrittleStarMorphology(
            specification=morphology_specification
            )
    return morphology

def create_arena(
        arena_configuration: AquariumArenaConfiguration
        ) -> MJCFAquariumArena:
    arena = MJCFAquariumArena(
            configuration=arena_configuration
            )
    return arena

def create_environment(
        morphology_specification: BrittleStarMorphologySpecification,
        arena_configuration: AquariumArenaConfiguration,
        environment_configuration: MuJoCoEnvironmentConfiguration,
        backend: str
        ) -> DualMuJoCoEnvironment:
    assert backend in ["MJC", "MJX"], "Please specify a valid backend; Either 'MJC' or 'MJX'"

    morphology = create_morphology(
            morphology_specification=morphology_specification
            )
    arena = create_arena(
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


def full_mjcf_configurations(
        morph_cfg,
        arena_cfg,
        env_cfg
        ):
    # specifying morphology
    morphology_specification = default_brittle_star_morphology_specification(
            num_arms=len(morph_cfg["arm_setup"]),
            num_segments_per_arm=morph_cfg["arm_setup"],
            use_p_control=(morph_cfg["joint_control"] == 'position'),
            use_torque_control=(morph_cfg["joint_control"] == 'torque')
            )

    # specifying arena
    arena_configuration = AquariumArenaConfiguration(
            size=arena_cfg["arena_size"],
            sand_ground_color=arena_cfg["sand_ground_color"],
            attach_target=(env_cfg["reward_type"]=="target"),
            wall_height=1.5,
            wall_thickness=0.1
            )

    # specifying environment: CHOOSE 1
    assert env_cfg["reward_type"] in (["distance","target","light"]), "reward_type must be one of 'distance', 'target', 'light'"

    if env_cfg["reward_type"] == "distance":
        environment_configuration = BrittleStarUndirectedLocomotionEnvironmentConfiguration(
            # If this value is > 0 then we will add randomly sampled noise to the initial joint positions and velocities 
            joint_randomization_noise_scale=env_cfg["joint_randomization_noise_scale"],
            render_mode="rgb_array",  # Visualization mode 
            simulation_time=env_cfg["simulation_time"],  # Number of seconds per episode 
            num_physics_steps_per_control_step=env_cfg["num_physics_steps_per_control_step"],  # Number of physics substeps to do per control step 
            time_scale=2,    # Integer factor by which to multiply the original physics timestep of 0.002,
            camera_ids=[0, 1],   # Which camera's to render (all the brittle star environments contain 2 cameras: 1 top-down camera and one close-up camera that follows the brittle star),
            render_size=(480, 640),  # Resolution to render with ((height, width) in pixels)
            color_contacts=env_cfg["color_contacts"]
            )

    elif env_cfg["reward_type"] == "target":
        environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
            # Distance to put our target at (targets are spawned on a circle around the starting location with this given radius).
            target_distance=env_cfg["target_distance"],
            joint_randomization_noise_scale=env_cfg["joint_randomization_noise_scale"],
            render_mode="rgb_array",
            simulation_time=env_cfg["simulation_time"],
            num_physics_steps_per_control_step=env_cfg["num_physics_steps_per_control_step"],
            time_scale=2,
            camera_ids=[0, 1],
            render_size=(480, 640),
            target_position=env_cfg["target_position"],
            color_contacts=env_cfg["color_contacts"]
            )

    elif env_cfg["reward_type"] == "light":
        environment_configuration = BrittleStarLightEscapeEnvironmentConfiguration(
            joint_randomization_noise_scale=env_cfg["joint_randomization_noise_scale"],
            light_perlin_noise_scale=env_cfg["light_perlin_noise_scale"],
            # If this value is > 0, we will add perlin noise to the generated light map. Otherwise, the light map is a simple linear gradient.
            # Please only provide integer factors of 200.
            render_mode="rgb_array",
            simulation_time=env_cfg["simulation_time"],
            num_physics_steps_per_control_step=env_cfg["num_physics_steps_per_control_step"],
            time_scale=2,
            camera_ids=[0, 1],
            render_size=(480, 640),
            color_contacts=env_cfg["color_contacts"]
            )
    
    return morphology_specification, arena_configuration, environment_configuration





