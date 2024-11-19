from typing import Union, Sequence, Type
import chex

import jax
from jax import numpy as jnp

from evosax import ParameterReshaper

from bsc_utils.BrittleStarEnv import EnvContainer
from bsc_utils.controller.base import NNController, ExplicitMLP
from bsc_utils.controller.hebbian import HebbianController


class DecentralisedController():
    """
    Initialise:
    - EnvContainer: the environment setup with configuration on for which the decentralised
                    controller will be used
    - parallel_dim (def=1): number of parallel environments to simulate. Used in initialisation
                            of arm states and central reservoir.

    Decentralised Controllers are constructed on Brittle Stars with 5 identical arms
    in the undamaged condition. It keeps track of all the internal states of the sensors,
    actuators, neural network kernels and biases, and the central reservoir.

    When damage is inflicted by removing segments, the neural network does not change.
    Only sensors (and possibly actuator) outputs are set to zero.
    """

    def __init__(
            self,
            env_container: EnvContainer,
            parallel_dim: int = 1
    ):
        assert env_container.config["controller"]["decentralized"]["decentralized_on"],\
            "Attempted to initialize the decentralized controller, but in the config file decentralized_on was set to False"
        
        assert all(element == env_container.config["morphology"]["arm_setup"][0] for element in env_container.config["morphology"]["arm_setup"]),\
            "For the decentralized controller, all arms in the undamaged condition must be identical"
        
        self._config = env_container.config
        self._env_container = env_container
        self._parallel_dim = parallel_dim

        self.reset_central_reservoir()
        self.reset_embed_output_layers()
        self.reset_arm_states()
    

    def reset_embed_output_layers(self):
        """
        Generates self._embed_layers and self._output_layers attributes with the structure
        [#sensors_per_arm, #hidden_nodes, #actuators_per_arm]
        E.g. when arm with 5 segments senses body contact, arm positions, arm forces and target direction allignment
             20 embed layer, 2 hidden layers van 64 neuronen:
        self._embed_layers = [26, 64, 64, 20]
        self._output_layers = [100, 64, 64, 10] (10 outputs, 100 in embedding layer)
        """
        sensory_input_dim = 0
        num_segm_per_arm = self._config["morphology"]["arm_setup"][0]

        segm_sensors_2 = ['joint_position', 'joint_velocity', 'joint_actuator_force'] # every segment has 2 sensors: ip and oop
        segm_sensors_1 = ['segment_contact', 'segment_light_intake'] # every segment has 1 sensor


        for sensor in self._config["environment"]["sensor_selection"]:
            if sensor in segm_sensors_2:
                sensory_input_dim += 2*num_segm_per_arm
            if sensor in segm_sensors_1:
                sensory_input_dim += 1*num_segm_per_arm
            if sensor == 'unit_xy_direction_to_target' and self._config["environment"]["reward_type"] == 'target':
                sensory_input_dim += 1 # just a factor for the allignment

        central_reservoir_dim = sum(jax.tree.map(lambda x: jnp.shape(x)[-1], self._central_reservoir).values())
        actuator_output_dim = num_segm_per_arm * 2 # always ip and oop actuator

        self._embed_layers = [sensory_input_dim] + self._config["controller"]["decentralized"]["embed_hidden_layers"]\
            + [self._config["controller"]["decentralized"]["embedding_dim_per_arm"]]
        self._output_layers = [central_reservoir_dim] + self._config["controller"]["decentralized"]["output_hidden_layers"]\
            + [actuator_output_dim]    


    def reset_arm_states(
            self,
    ):
        """
        The structure of the dict is something like:

        arms_state = 
        {'arm_0_embed':
            {'inputs': jnp.array,
            'layers_0': 
                {'kernel': jnp.array,
                'bias': jnp.array,
                'output': }
        arm_0_output: {...},
        arm_1_embed: {...},
        ...}

        The dimension of the arrays above is [#par_dim, #timesteps, 1D or 2D]
        - # parallel_dim: first dim related to all the parallel environments. This allows vmapping in training.
                          in analysis, this is by default set to 1
        - # timesteps: the number of timesteps that have already passed in the episode.

        The first data entry is the one as provided by this reset. (all zeroes)
        """
        arm_setup = self._config["morphology"]["arm_setup"]
        arm_states = {}
        
        for i in range(len(self._config["morphology"]["arm_setup"])):
            arm_states[f"arm_{i}_embed"] = {}
            arm_states[f"arm_{i}_embed"]["inputs"] = jnp.zeros((self._parallel_dim, 1, self._embed_layers[0]))
            for j in range(len(self._embed_layers)-1):
                arm_states[f"arm_{i}_embed"][f"layers_{j}"] = {}
                arm_states[f"arm_{i}_embed"][f"layers_{j}"]["kernel"] = jnp.zeros((self._parallel_dim, 1,self._embed_layers[j], self._embed_layers[j+1]))
                arm_states[f"arm_{i}_embed"][f"layers_{j}"]["bias"] = jnp.zeros((self._parallel_dim, 1,self._embed_layers[j+1]))
                arm_states[f"arm_{i}_embed"][f"layers_{j}"]["output"] = jnp.zeros((self._parallel_dim, 1,self._embed_layers[j+1]))

            arm_states[f"arm_{i}_output"] = {}
            arm_states[f"arm_{i}_output"]["inputs"] = jnp.zeros((self._parallel_dim, 1, self._output_layers[0]))
            for j in range(len(self._output_layers)-1):
                arm_states[f"arm_{i}_output"][f"layers_{j}"] = {}
                arm_states[f"arm_{i}_output"][f"layers_{j}"]["kernel"] = jnp.zeros((self._parallel_dim, 1,self._output_layers[j], self._output_layers[j+1]))
                arm_states[f"arm_{i}_output"][f"layers_{j}"]["bias"] = jnp.zeros((self._parallel_dim, 1,self._output_layers[j+1]))
                arm_states[f"arm_{i}_output"][f"layers_{j}"]["output"] = jnp.zeros((self._parallel_dim, 1,self._output_layers[j+1]))
            
        self._arm_states = arm_states
    

    def reset_central_reservoir(
            self,
    ):
        """
        resets the reservoir to be zero
        The structure of the central reservoir is:
        {arm_0: nd.array(),
         arm_1: ...,
         ...
         (optional) body_sensors: nd.array()
        }
        where every array has dimension [#parallel_dim, #timesteps, embedding per arm OR num_body_sensors]:
        - # parallel_dim: allows possibly vmapping during training across popsize.
        - # timesteps: number of timesteps that have already passed
        - embedding per arm: for the arms, the central reservoir receives this number of values
        - num_body_sensors: if wanted, this can be enabled to pass the body sensor information directly to the central reservoir.
        """
        body_sensors_3D = ['disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity'] # sensors are related to body
        body_sensors_2D = ['unit_xy_direction_to_target']
        body_sensors_1D = ['xy_distance_to_target']
        num_body_sensors = 0

        central_reservoir = {}
        for i in range(len(self._config["morphology"]["arm_setup"])):
            central_reservoir[f"arm_{i}"] = jnp.zeros((self._parallel_dim, 1, self._config["controller"]["decentralized"]["embedding_dim_per_arm"]))
        
        # # Currently don't include the use of body centered based metrics.
        # for sensor in self._config["environment"]["sensory_selection"]:
        #     if sensor in body_sensors_3D:
        #         num_body_sensors += 3
        #     elif sensor in body_sensors_2D:
        #         num_body_sensors += 2
        #     elif sensor in body_sensors:
        #         num_body_sensors += 1
        # central_reservoir["body_sensors"] = jnp.zeros((self._parallel_dim, 1, num_body_sensors))

        self._central_reservoir = central_reservoir


    @property
    def config(self) -> dict:
        """
        Returns the config file that is contained in the EnvContainer
        """
        return self._config

    @property
    def arm_states(self) -> dict:
        """
        Arm state is a dict containing the represenation of every arm for the complete history of
        - Observations made so far
        - NN-layers (dict with kernel, bias and outputs)

        Every leaf has as a first dimension the number of timesteps that have been simulated so far
        """
        return self._arm_states
    
    @property
    def central_reservoir(self) -> dict:
        """
        Central reservoir is a dict containing all the information in the central reservoir for
        the complete history of the agent:
        - the embedding per arm

        structured like:
        central_reservoir = { arm_1: nd.array(#timesteps_passed, embedding_dim_per_arm), arm_2: [...], ... }
        """
        return self._central_reservoir
    
    @property
    def embed_layers(self) -> list:
        return self._embed_layers
    
    @property
    def output_layers(self) -> list:
        return self._output_layers
    
    @property
    def parallel_dim(self) -> int:
        return self._parallel_dim


    def update_model(
            self,
            controller = HebbianController
    ):
        """
        Will create a model for every arm. All these models are stored in a dict.
        """
        
        if controller == HebbianController:
            arm_models = {}
            for i in range(len(self._config["morphology"]["arm_setup"])):
                arm = {}
                arm["embed"] = HebbianController(self._env_container)
                arm["embed"].update_model(layer_architecture=self._embed_layers)
                arm["output"] = HebbianController(self._env_container)
                arm["output"].update_model(layer_architecture=self._output_layers)
                arm_models[f"arm_{i}"] = arm

        else:
            raise NotImplementedError("the specified controller type has not been implemented. Consider using a HebbianController")

        self._arm_models = arm_models

    def get_policy_params_example(self) -> dict:
        """
        Based on the the embedding and output layer of 1 arm, generate empty policy params examples
        These policy params will contain the Hebbian learning rule dimension.
        
        Note: the structure of the dict is
        {embed:
            params:
                layers_0:
                    kernel:...,
                    bias:...
                layers_1:
                    ...
         output:
            ...
        }
        """
        embed_policy_params_example = self._arm_models["arm_0"]["embed"].get_policy_params_example()
        output_policy_params_example = self._arm_models["arm_0"]["output"].get_policy_params_example()
        embed_output_policy_params_example = {}
        embed_output_policy_params_example["embed"] = embed_policy_params_example
        embed_output_policy_params_example["output"] = output_policy_params_example

        return embed_output_policy_params_example
    

    def update_parameter_reshaper(self):
        """
        Instantiates attribute self.parameter_reshaper (as an object of evosax.ParameterReshaper class)
        """
        assert self._arm_models, "No model has been instantiated yet. Use method update_model"
        policy_params_example = self.get_policy_params_example()
        self.parameter_reshaper = ParameterReshaper(policy_params_example)

    
    def get_parameter_reshaper(self) -> ParameterReshaper:
        """
        Returns the parameter_reshaper that should be used by the evosax OpenAI algorithm
        """
        assert self.parameter_reshaper, "first instantiate the self.parameter_reshaper attribute by calling update_parameter_reshaper method"
        return self.parameter_reshaper
    

    def reset_synapse_strengths_unif(
            self,
            rng: chex.PRNGKey,
            parallel_dim: int = None, # either popsize during training or 1 or multiple parallel environments during the evaluation after training.
            minval: float = -0.1,
            maxval: float = 0.1
    ):
        """
        Goes over all arms and resets their synapse strengths
        inputs:
        - rng: random number generator for the uniform initialisation
        - parallel_dim: since thise synapses are feeded to a vmapped apply function,
            the first dimension must represent the parallel dimensions to map across
            If no parallel_dim is provided, the one used in the initialisation is used.
        - minval: lower boundary of uniform initialisation
        - maxval: upper boundary of uniform initialisation
        """
        if parallel_dim == None:
            parallel_dim = self._parallel_dim

        for i in range(len(self._config["morphology"]["arm_setup"])):
            for j in range(len(self._embed_layers)-1):
                rng, rng_unif = jax.random.split(rng, 2)
                dim = self._arm_states[f"arm_{i}_embed"][f"layers_{j}"]["kernel"].shape
                self._arm_states[f"arm_{i}_embed"][f"layers_{j}"]["kernel"] =\
                    jax.random.uniform(rng_unif, shape=dim, minval = minval, maxval = maxval)
            for k in range(len(self._output_layers)-1):
                rng, rng_unif = jax.random.split(rng, 2)
                dim = self._arm_states[f"arm_{i}_output"][f"layers_{k}"]["kernel"].shape
                self._arm_states[f"arm_{i}_output"][f"layers_{k}"]["kernel"] =\
                    jax.random.uniform(rng_unif, shape=dim, minval = minval, maxval = maxval)


    def update_learning_rules(
            self,
            learning_rules: Union[dict, chex.Array]
    ):
        """
        If learning_rules are an array, the ParameterReshaper associated to the controller
        is applied to this array first.
        If the learning_rules are already a dict, they are immediately applied to the respective arms.

        The dict has format:
        {embed:
            params:
                layers_0:
                    kernel:...,
                    bias:...
                layers_1:
                    ...
         output:
            ...
        }
        """

        if isinstance(learning_rules, chex.Array):
            learning_rules = self.parameter_reshaper.reshape(learning_rules)
        
        for i in range(len(self._config["morphology"]["arm_setup"])):
            self._arm_models[f"arm_{i}"]["embed"].update_learning_rules(learning_rules["embed"])
            self._arm_models[f"arm_{i}"]["output"].update_learning_rules(learning_rules["output"])

    
    # def arm_states_timestep_update(
    #         self,
    #         sensory_input: chex.Array,
    #         synapse_strengths: dict,
    #         learning_rules: dict,
    #         neuron_activities: Sequence[chex.Array]

    # ):
    #     raise NotImplementedError
    

    def apply(
            self,
            sensory_input: chex.Array,
            
    ):
        raise NotImplementedError



    




