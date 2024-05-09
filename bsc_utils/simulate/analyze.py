import chex
import numpy as np
from typing import Union
import jax
from jax import numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt

from evosax import ParameterReshaper

from bsc_utils.BrittleStarEnv import EnvContainer
from bsc_utils.controller.base import NNController
from bsc_utils.controller.hebbian import HebbianController
from bsc_utils.visualization import post_render, change_alpha, move_camera, generate_timestep_joint_angle_plot_data, \
    plot_ip_oop_joint_angles, save_video_from_raw_frames
from bsc_utils.damage import pad_sensory_input, select_actuator_output, check_damage
from bsc_utils.simulate.base import cost_step_during_rollout, penal_step_during_rollout
from bsc_utils.evolution import efficiency_from_reward_cost, fitness_from_stacked_data


class Simulator(EnvContainer):
    def __init__(self,
                 config: dict
    ):
        super().__init__(config)

    
    def update_policy_params_flat(self, policy_params_flat): # This function is probably obsolete, as this functionality moved
        # to the NNController class as the method "update_policy_params"
        """
        Provide in a simple numpy array format (not the ParamReshaped version)
        Multiple policy params can be rendered in parallel
        These can be the policy params of a Hebbian network (Learning Rules) or from a static network (Synapse strengths)
        """
        self.policy_params_flat = policy_params_flat

    def update_nn_controller(
            self,
            nn_controller: Union[NNController, HebbianController]
    ):
        """
        Before updating nn_controller, try to make sure that it has a model attribute (method update_model)
        and parameter_reshaper attribute (method update_parameter_reshaper)
        """
        self.nn_controller = nn_controller


    def generate_episode_data_undamaged(
            self,
            rng: chex.PRNGKey
    ):
        assert self.env, "First instantiate an undamaged environment using the generate_env method"
        self._generate_episode_data(rng, damage = False)

    def generate_episode_data_damaged(
            self,
            rng: chex.PRNGKey
    ):
        assert self.env_damage, "First instantiate a damaged environment using the generate_env_damaged method"
        check_damage(self.config["morphology"]["arm_setup"], self.config["damage"]["arm_setup_damage"])
        self._generate_episode_data(rng, damage = True)


    def get_episode_reward(self):
        assert np.any(self.rewards), "First run an episode using generate_episode_data_(un)damaged"
        return jnp.sum(self.rewards, axis = -1)

    def get_episode_cost(self):
        assert self.observations, "First run an episode using generate_episode_data_(un)damaged"
        _cost_step = cost_step_during_rollout(self.observations, self.config["evolution"]["cost_expr"])
        return jnp.sum(_cost_step, axis = -1) # return array of costs of complete morphology over complete episode for every parallel episode
    
    def get_episode_penalty(self):
        assert self.observations, "First run an episode using generate_episode_data_(un)damaged"
        _penal_step = penal_step_during_rollout(self.observations, self.config["evolution"]["penal_expr"])
        return jnp.sum(_penal_step, axis = -1)
    
    def get_episode_efficiency(self):
        assert self.observations, "First run an episode using generate_episode_data_(un)damaged"
        _reward = self.get_episode_reward()
        _cost = self.get_episode_cost()
        efficiency = efficiency_from_reward_cost(_reward, _cost, self.config["evolution"]["efficiency_expr"])
        return efficiency
    
    def get_episode_fitness(self):
        assert self.observations, "First run an episode using generate_episode_data_(un)damaged"
        _reward = self.get_episode_reward()
        _cost = self.get_episode_cost()
        _penalty = self.get_episode_penalty()
        _stack = (_reward, _cost, _penalty)
        fitness = fitness_from_stacked_data(_stack, self.config["evolution"]["efficiency_expr"])
        return fitness
    
    def get_ip_oop_joint_angles_plot(
            self,
            file_path: str = None,
            show_image: bool = False
    ):
        """
        filepath should end in .png, .jpg, ...
        """
        assert np.any(self.joint_angles_ip), "First run an episode using generate_episode_data_(un)damaged"
        fig, axes = plot_ip_oop_joint_angles(self.joint_angles_ip, self.joint_angles_oop)
        if file_path:
            fig.savefig(file_path)
        if show_image:
            fig.show()

    def get_episode_video(
            self,
            file_path: str = None,
            playback_speed: float = 1.0
    ):
        """
        filepath should end in .mp4
        """
        assert np.any(self.frames), "First run an episode using generate_episode_data_(un)damaged"
        if (self.config["environment"]["render"]["render_size"][0] <= 1440) and (self.config["environment"]["render"]["render_size"][1] <= 1920): # max size can be 1080, otherwise files get too big
            _fps = int(1/self.environment_configuration.control_timestep)
            _fps *= playback_speed
            if file_path:
                save_video_from_raw_frames(self.frames, _fps, file_path)
        else:
            print("the rendersize provided was too big to attempt rendering: choose rendersize [ 1440, 1920 ] (1080p) or smaller")
    
    def get_increasing_opacity_image(
            self,
            number_of_frames: int,
            file_path: str = None,
            show_image: bool = False
    ):
        """
        filepath should end in .png, .jpg, ...
        """
        assert np.any(self.background_frame), "First run an episode using generate_episode_data_(un)damaged"
        merged_frame = self.background_frame
        selected_frames = self.brittle_star_frames[::len(self.brittle_star_frames)//number_of_frames]
        for i, brittle_star_frame in enumerate(selected_frames):
            # selecting specific frames to get only the brittle star
            ####################################################
            if i == 0:
                tmp_img = Image.fromarray(brittle_star_frame, 'RGB')
                tmp_img.save("C:\\Users\\Matthias\\OneDrive - UGent\\Documents\\DOCUMENTEN\\3. Thesis\\BSC\\Images\\tmp\\singular frame undamaged.png")
            ####################################################
            alpha = i / len(selected_frames)

            # Boolean mask with only the brittle star
            brittle_star_mask = brittle_star_frame != [255, 255, 255]

            # Alpha blending of brittle star with current merge
            merged_brittle_star = (1 - alpha) * merged_frame + alpha * brittle_star_frame

            # Replace the brittle star's pixels --> all the other pixels remain the same
            merged_frame[brittle_star_mask] = merged_brittle_star[brittle_star_mask]

        img = Image.fromarray(merged_frame, 'RGB')
        if file_path:
            img.save(file_path)
        if show_image:
            img.show()

    
    def _generate_episode_data(
            self,
            rng: chex.PRNGKey,
            damage: bool = False
    ):
        """
        Generates attributes:
        - frames
        - joint_angles_ip
        - joint_angels_oop
        - background_frame
        - brittle_star_frames
        - stacked_observations: trees with leaf dims: [n, m, t]: n = parallel envs, m = sensor dim, t = timesteps during rollout
        - stacked rewards: array with dims [n, t]: n = parallel envs, t = timesteps during rollout
        """
        assert self.nn_controller, "No nn_controller has been provided yet using the update_nn_controller method"

        if damage:
            _env = self.env_damage
        else:
            _env = self.env

        _policy_params = self.nn_controller.get_policy_params()
        _NUM_MJX_ENVIRONMENTS = _policy_params["params"]["layers_0"]["kernel"].shape[0]

        rng, _vectorized_env_rng = jax.random.split(rng, 2)
        _vectorized_env_rng = jnp.array(jax.random.split(_vectorized_env_rng, _NUM_MJX_ENVIRONMENTS))

        _vectorized_env_step = jax.jit(jax.vmap(_env.step))
        _vectorized_env_reset = jax.jit(jax.vmap(_env.reset))

        _vectorized_controller_apply = jax.jit(jax.vmap(self.nn_controller.apply))

        _vectorized_env_state = _vectorized_env_reset(rng=_vectorized_env_rng)

        # reset for Hebbian learning:
        if self.config["controller"]["hebbian"]:
            rng, _rng_ss_reset = jax.random.split(rng, 2)
            self.nn_controller.reset_synapse_strengths_unif(_rng_ss_reset, parallel_dim=_NUM_MJX_ENVIRONMENTS)
            self.nn_controller.reset_neuron_activities(parallel_dim=_NUM_MJX_ENVIRONMENTS)
            _synapse_strengths = self.nn_controller.get_synapse_strengths()
            _neuron_activities = self.nn_controller.get_neuron_activities()

        # visualisation
        _frames = []
        _joint_angles_ip = []
        _joint_angles_oop = []
        _brittle_star_frames = []

        _env_state_background = move_camera(state=_vectorized_env_state)
        _env_state_background = change_alpha(state = _env_state_background, brittle_star_alpha=0.0, background_alpha=1.0)
        _background_frame = post_render(
            _env.render(state=_env_state_background),
            _env.environment_configuration
            )
        
        t = 0
        while not jnp.any(_vectorized_env_state.terminated | _vectorized_env_state.truncated):
            
            _sensory_input = jnp.concatenate(
                [_vectorized_env_state.observations[label] for label in self.config["environment"]["sensor_selection"]],
                axis = 1
            )

            if damage:
                _sensory_input = pad_sensory_input(
                    _sensory_input,
                    self.config["morphology"]["arm_setup"],
                    self.config["damage"]["arm_setup_damage"],
                    self.config["environment"]["sensor_selection"]
                    )


            if damage:
                arms_joint_angles_plot = self.config["damage"]["arm_setup_damage"]
            else:
                arms_joint_angles_plot = self.config["morphology"]["arm_setup"]

            _joint_angles_ip_t, _joint_angles_oop_t = generate_timestep_joint_angle_plot_data(arms_joint_angles_plot, _vectorized_env_state)
            _joint_angles_ip.append(_joint_angles_ip_t)
            _joint_angles_oop.append(_joint_angles_oop_t)


            # action hebbian plastic nn or static nn:
            if self.config["controller"]["hebbian"]:
                # apply a Hebbian control: updates synapse strengths using learning rules, yields action and neuron activities
                _action, _synapse_strengths, _neuron_activities = _vectorized_controller_apply(
                                                                        sensory_input=_sensory_input,
                                                                        synapse_strengths=_synapse_strengths,
                                                                        learning_rules=_policy_params, # learning rules in case of hebbian
                                                                        neuron_activities=_neuron_activities
                                                                        )

            else:
                # apply a static control: just yields action and neuron activities
                _action, _neuron_activities = _vectorized_controller_apply(
                                                                        sensory_input=_sensory_input,
                                                                        synapse_strengths=_policy_params # synapse strengths in case of not hebbian
                                                                        )

            if damage:
                _action = select_actuator_output(_action, self.config["morphology"]["arm_setup"], self.config["damage"]["arm_setup_damage"])
            
            _vectorized_env_state = _vectorized_env_step(state=_vectorized_env_state, action=_action)

            if t == 0:
                _observations = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis = -1), _vectorized_env_state.observations)
                _rewards = jnp.expand_dims(_vectorized_env_state.reward, axis = -1)
            else:
                _observations = jax.tree_util.tree_map(
                    lambda x, y: jnp.concatenate(
                        [x, jnp.expand_dims(y, axis = -1)],
                        axis=-1),
                    _observations, _vectorized_env_state.observations)
                _rewards = jnp.concatenate(
                    [_rewards, jnp.expand_dims(_vectorized_env_state.reward, axis = -1)],
                    axis = -1)


            _frames.append(
                post_render(
                    _env.render(state=_vectorized_env_state),
                    _env.environment_configuration
                    )
                )
            
            _env_state_brittle_star = move_camera(state=_vectorized_env_state)
            _env_state_brittle_star = change_alpha(state = _env_state_brittle_star, brittle_star_alpha=1.0, background_alpha=0.0)
            _brittle_star_frames.append(
                post_render(
                _env.render(state=_env_state_brittle_star),
                _env.environment_configuration
                )
            )

            t += 1
        
        self.frames = _frames
        self.joint_angles_ip = _joint_angles_ip
        self.joint_angles_oop = _joint_angles_oop
        self.brittle_star_frames = _brittle_star_frames
        self.background_frame = _background_frame
        self.observations = _observations
        self.rewards = _rewards






# example use:
if __name__ == "__main__":

    import os
    import jax
    from jax import numpy as jnp

    from evosax import ParameterReshaper

    from bsc_utils.miscellaneous import load_config_from_yaml
    from bsc_utils.simulate.analyze import Simulator
    from bsc_utils.controller.base import NNController, ExplicitMLP

    rng = jax.random.PRNGKey(0)

    # NOTICE: RUNNING BOTH SIMULATIONS ON LARGE MEMORY GIVES PROBLEMS SAVING THE IMAGES     

    VIDEO_DIR = os.environ["VIDEO_DIR"]
    IMAGE_DIR = os.environ["IMAGE_DIR"]
    POLICY_PARAMS_DIR = os.environ["POLICY_PARAMS_DIR"]
    RUN_NAME = os.environ["RUN_NAME"]

    trained_policy_params_flat = jnp.load(POLICY_PARAMS_DIR + RUN_NAME + ".npy")
    config = load_config_from_yaml(POLICY_PARAMS_DIR + RUN_NAME + ".yaml")

    ####################################################################################
    # finutune episode simulation
    simulate_undamaged = False
    simulate_damaged = True
    config["damage"]["arm_setup_damage"] = [5,0,5,5,5]
    config["arena"]["sand_ground_color"] = True
    config["environment"]["render"] = {"render_size": [ 480, 640 ], "camera_ids": [ 0, 1 ]} # only static aquarium camera camera [ 0 ], otherwise: "camera_ids": [ 0, 1 ]
                                # choose ratio 3:4 --> [ 480, 640 ], [ 720, 960 ], [ 960, 1280 ] (720p), [ 1440, 1920 ] (1080p), [ 3072, 4069 ] (HDTV 4k)
    # config["evolution"]["penal_expr"] = "nopenal"
    # config["evolution"]["efficiency_expr"] = config["evolution"]["fitness_expr"]
    ####################################################################################



    simulator = Simulator(config)
    simulator.generate_env()
    simulator.generate_env_damaged()
    observation_space_dim, actuator_space_dim = simulator.get_observation_action_space_info()
    print(f"""
    observation_space_dim = {observation_space_dim}
    actuator_space_dim = {actuator_space_dim}
    """)

    nn_controller = NNController(simulator)
    nn_controller.update_model(ExplicitMLP)
    nn_controller.update_parameter_reshaper() # as the model is already defined and the environmetns are available from the environment container and config files in simulator in the simulator
    # param_reshaper = nn_controller.get_parameter_reshaper() # --> if you would want to do stuff with this parameter_reshaper.

    simulator.update_policy_params_flat(trained_policy_params_flat)
    simulator.update_nn_controller(nn_controller)


    if simulate_undamaged:
        print("simulation of single episode started: Undamaged")
        rng, rng_episode = jax.random.split(rng, 2)
        simulator.generate_episode_data_undamaged(rng_episode)
        print("simulation of single episode finished: Undamaged")

        reward = simulator.get_episode_reward()
        cost  = simulator.get_episode_cost()
        penalty = simulator.get_episode_penalty()
        efficiency = simulator.get_episode_efficiency()
        fitness = simulator.get_episode_fitness()
        simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + RUN_NAME + ".png")
        simulator.get_increasing_opacity_image(number_of_frames=8, file_path=IMAGE_DIR + RUN_NAME + " OPACITY.png")
        simulator.get_episode_video(file_path = VIDEO_DIR + RUN_NAME + ".mp4", playback_speed=0.5)


        print(f"""
        reward = {reward}
        cost = {cost}
        penalty = {penalty}
        efficiency = {efficiency}
        fitness = {fitness}
        """)

    if simulate_damaged:
        print("simulation of single episode started: Damaged")
        rng, rng_episode = jax.random.split(rng, 2)
        simulator.generate_episode_data_damaged(rng_episode)
        print("simulation of single episode finished: Damaged")

        reward_damage = simulator.get_episode_reward()
        cost_damage  = simulator.get_episode_cost()
        penalty_damage = simulator.get_episode_penalty()
        efficiency_damage = simulator.get_episode_efficiency()
        fitness_damage = simulator.get_episode_fitness()
        simulator.get_ip_oop_joint_angles_plot(file_path = IMAGE_DIR + RUN_NAME + " DAMAGE.png")
        simulator.get_increasing_opacity_image(number_of_frames=8, file_path=IMAGE_DIR + RUN_NAME + "OPACITY DAMAGE.png")
        simulator.get_episode_video(file_path = VIDEO_DIR + RUN_NAME + " DAMAGE.mp4", playback_speed=0.5)


        print(f"""
        reward_damage = {reward_damage}
        cost_damage = {cost_damage}
        penalty_damage = {penalty_damage}
        efficiency_damage = {efficiency_damage}
        fitness_damage = {fitness_damage}
        """)

    if simulate_undamaged and simulate_damaged:
        print(f"""
        reward = {reward} - reward_damage = {reward_damage}
        cost = {cost} - cost_damage = {cost_damage}
        penalty = {penalty} - penalty_damage = {penalty_damage}
        efficiency = {efficiency} - efficiency_damage = {efficiency_damage}
        fitness = {fitness} - fitness_damage = {fitness_damage}
        """)


    simulator.clear_envs()

