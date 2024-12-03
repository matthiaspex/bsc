from typing import Union, Sequence, Type, Literal, Tuple, Optional
import chex
import sys

import jax
from jax import numpy as jnp

from evosax import ParameterReshaper

from bsc_utils.BrittleStarEnv import EnvContainer
from bsc_utils.controller.base import NNController
from bsc_utils.controller.hebbian import HebbianController
from bsc_utils.miscellaneous import check_sensor_selection_order,\
    calculate_arm_target_allignment_factors


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
            parallel_dim: int
    ):
        assert env_container.config["controller"]["decentralized"]["decentralized_on"],\
            "Attempted to initialize the decentralized controller, but in the config file decentralized_on was set to False"
        
        assert all(element == env_container.config["morphology"]["arm_setup"][0] for element in env_container.config["morphology"]["arm_setup"]),\
            "For the decentralized controller, all arms in the undamaged condition must be identical"
        
        self._config = env_container.config
        self._env_container = env_container
        self._parallel_dim = parallel_dim

        self.reset_embed_output_layers()
        # for the initialisation, the reset does not really matter
        rng_init = jax.random.PRNGKey(seed=0)
        self.reset_states(rng=rng_init)
    

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

        body_sensors_3D = ['disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity'] # sensors are related to body
        body_sensors_2D = ['unit_xy_direction_to_target']
        body_sensors_1D = ['xy_distance_to_target']
        num_body_sensors = 0
        central_reservoir_dim = 0

        if self._config["controller"]["decentralized"]["explicit_body_sensors"]:
            for sensor in self._config["environment"]["sensory_selection"]:
                if sensor in body_sensors_3D:
                    num_body_sensors += 3
                elif sensor in body_sensors_2D:
                    num_body_sensors += 2
                elif sensor in body_sensors_1D:
                    num_body_sensors += 1

        central_reservoir_dim += num_body_sensors
        central_reservoir_dim += len(self._config["morphology"]["arm_setup"]) * self._config["controller"]["decentralized"]["embedding_dim_per_arm"]

        for sensor in self._config["environment"]["sensor_selection"]:
            if sensor in segm_sensors_2:
                sensory_input_dim += 2*num_segm_per_arm
            if sensor in segm_sensors_1:
                sensory_input_dim += 1*num_segm_per_arm
            if sensor == 'unit_xy_direction_to_target' and self._config["environment"]["reward_type"] == 'target':
                sensory_input_dim += 1 # just a factor for the allignment

        actuator_output_dim = num_segm_per_arm * 2 # always ip and oop actuator

        self._embed_layers = [sensory_input_dim] + self._config["controller"]["decentralized"]["embed_hidden_layers"]\
            + [self._config["controller"]["decentralized"]["embedding_dim_per_arm"]]
        self._output_layers = [central_reservoir_dim] + self._config["controller"]["decentralized"]["output_hidden_layers"]\
            + [actuator_output_dim]   
        
        self._num_body_sensors = num_body_sensors 
        self._central_reservoir_dim = central_reservoir_dim



    def reset_states(
            self,
            rng: chex.PRNGKey,
            **kwargs
    ):
        """
        input:
        - rng: for the uniform initialisation of the synaptic weights
        - **kwargs: optional arguments for the reset_synapse_strengths
            -> parallel_dim = None (by default take the parallel dim from config file)
            -> minval = -0.1 (by default synapse strengths minimum of -0.1 at initialization)
            -> maxval = +0.1 (by deafult synapse strengths maximum of 0.1 at initialization)
        The structure of the dict is something like:

        arms_state = 
        {'arm_0_embed':
            {'inputs': jnp.array,
            'layers_0': 
                {'kernel': jnp.array,
                'bias': jnp.array,
                'output': }
         'arm_0_output': {...},
         'arm_1_embed': {...},
         ...,
         'central reservoir':
            {arm_0: nd.array(),
             arm_1: ...,
             ...,
             (optional) body_sensors: nd.array()
            }
        }

        The dimension of the arrays above is [#par_dim, #timesteps, 1D or 2D]
        - # parallel_dim: first dim related to all the parallel environments. This allows vmapping in training.
                          in analysis, this is by default set to 1
        - # timesteps: the number of timesteps that have already passed in the episode.
        - 1D or 2D array depends on: kernel, bias, input, output, arm embedding (central reservoir), body sensors, ...

        The first data entry is the one as provided by this reset. (all zeroes)
        """

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

        central_reservoir = {}
        for i in range(len(self._config["morphology"]["arm_setup"])):
            central_reservoir[f"arm_{i}"] = jnp.zeros((self._parallel_dim, 1, self._config["controller"]["decentralized"]["embedding_dim_per_arm"]))
        
        if self._num_body_sensors != 0:
            central_reservoir["body_sensors"] = jnp.zeros((self._parallel_dim, 1, self._num_body_sensors))

        arm_states["central_reservoir"] = central_reservoir

        #check if the central reservoir dim is indeed the same as calculated before
        assert sum(jax.tree.map(lambda x: jnp.shape(x)[-1], central_reservoir).values()) == self._central_reservoir_dim,\
        "Something went wrong in the calculation of the central reservoir dim in the 'reset_embed_output_layers' method"

        arm_states = self.reset_synapse_strengths_unif(states=arm_states, rng=rng, **kwargs)

        self._arm_states = arm_states
    

    @property
    def config(self) -> dict:
        """
        Returns the config file that is contained in the EnvContainer
        """
        return self._config

    @property
    def states(self) -> dict:
        """
        Arm state is a dict containing the represenation of every arm and central reservoir for the complete history of
        - Observations made so far
        - NN-layers (dict with kernel, bias and outputs)
        - central reservoir: embedding per arm and body sensor observations per arm

        Every leaf has as a first dimension the parallel dim (can be 1).
        Every leaf has as a second dimension number of timesteps that have been simulated so far
        """
        return self._arm_states
    
    
    @property
    def embed_layers(self) -> list:
        return self._embed_layers
    
    @property
    def output_layers(self) -> list:
        return self._output_layers
    
    @property
    def parallel_dim(self) -> int:
        return self._parallel_dim
    
    @property
    def model(self) -> tuple[HebbianController]:
        return (self._model_embed, self._model_output)
    

    @property
    def policy_params(self) -> dict:
        return self._policy_params
    
    @property
    def env_container(self) -> EnvContainer:
        return self._env_container
    

    def update_model(
            self,
            controller: Union[HebbianController, NNController]
    ):
        """
        Will create a model for the embed (afferent) neural network and the output (efferent) neural network
        """
        
        if controller == HebbianController:
            model_embed = HebbianController(self._env_container)
            model_embed.update_model(layer_architecture=self._embed_layers)
            model_output = HebbianController(self._env_container)
            model_output.update_model(layer_architecture=self._output_layers)

        elif controller == NNController:
            model_embed = NNController(self._env_container)
            model_embed.update_model(layer_architecture=self._embed_layers)
            model_output = NNController(self._env_container)
            model_output.update_model(layer_architecture=self._output_layers)
        else:
            raise NotImplementedError("the specified controller type has not been implemented. Consider using a HebbianController")

        self._model_embed = model_embed
        self._model_output = model_output

    def set_model(
            self,
            model: Tuple[Union[HebbianController, NNController]]
    ):
        self._model_embed = model[0]
        self._model_output = model[1]

    def set_states(
            self,
            states: dict
    ):
        self._arm_states = states

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
        embed_policy_params_example = self._model_embed.get_policy_params_example()
        output_policy_params_example = self._model_output.get_policy_params_example()
        embed_output_policy_params_example = {}
        embed_output_policy_params_example["embed"] = embed_policy_params_example
        embed_output_policy_params_example["output"] = output_policy_params_example

        return embed_output_policy_params_example
    

    def update_parameter_reshaper(self):
        """
        Instantiates attribute self.parameter_reshaper (as an object of evosax.ParameterReshaper class)
        """
        assert self._model_embed, "No model has been instantiated yet. Use method update_model"
        assert self._model_output, "No model has been instantiated yet. Use method update_model"
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
            states: dict,
            rng: Optional[chex.PRNGKey] = None,
            parallel_dim: int = None, # either popsize during training or 1 or multiple parallel environments during the evaluation after training.
            minval: float = -0.1,
            maxval: float = 0.1
    ) -> dict:
        """
        Goes over all arms and resets their synapse strengths
        inputs:
        - rng: random number generator for the uniform initialisation
        - parallel_dim: since thise synapses are feeded to a vmapped apply function,
            the first dimension must represent the parallel dimensions to map across
            If no parallel_dim is provided, the one used in the initialisation is used.
        - minval: lower boundary of uniform initialisation
        - maxval: upper boundary of uniform initialisation

        No side effects
        """
        # sometimes, the initialisation really doesn't matter, for instance when just creating dummies
        if parallel_dim == None:
            parallel_dim = self._parallel_dim

        for i in range(len(self._config["morphology"]["arm_setup"])):
            for j in range(len(self._embed_layers)-1):
                rng, rng_unif = jax.random.split(rng, 2)
                dim = states[f"arm_{i}_embed"][f"layers_{j}"]["kernel"].shape
                states[f"arm_{i}_embed"][f"layers_{j}"]["kernel"] =\
                    jax.random.uniform(rng_unif, shape=dim, minval = minval, maxval = maxval)
            for k in range(len(self._output_layers)-1):
                rng, rng_unif = jax.random.split(rng, 2)
                dim = states[f"arm_{i}_output"][f"layers_{k}"]["kernel"].shape
                states[f"arm_{i}_output"][f"layers_{k}"]["kernel"] =\
                    jax.random.uniform(rng_unif, shape=dim, minval = minval, maxval = maxval)
                
        return states


    def update_policy_params( # for this controller, this is basically update learning rules
            self,
            policy_params: Union[dict, chex.Array]
    ):
        """
        Stores the policy params into the respective learning rules of all the arms.

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
        assert self.parameter_reshaper, "first instantiate a ParameterReshaper by calling method update_parameter_reshaper"

        if isinstance(policy_params, chex.Array):
            policy_params = self.parameter_reshaper.reshape(policy_params)
        
        self._policy_params = policy_params

        self._model_embed.update_parameter_reshaper()
        self._model_output.update_parameter_reshaper()
        self._model_embed.update_policy_params(policy_params["embed"])
        self._model_output.update_policy_params(policy_params["output"])

    
    

    def apply(
            self,
            sensory_input: chex.Array,
            states: dict,
    ) -> Tuple[chex.Array, dict]:
        """
        input:
        - sensory_input: array with dim (popsize, sensory_input_dim)

        output:
        - actuation signal for the brittle star simulator

        Unlike other controllers, the apply function of this controller is not vmappable or jittable
        because of the specific way the internal states of the decentralised controller are stored
        in the arm_states attribute and central_reservoir attribute.
        
        Instead, this function launches the entire flow of information through the network
        when the sensory input is provided. After all, all information is already present
        in the controller to do this rollout.

        Internally, this function makes use of a vmapped apply of all the Hebbian subnetworks
        in every arm.

        Additionally, this function has the side effect of updating the internal states.
        Note: since this function allows side effects, it cannot be jitted.
        """

        states = self.store_sensory_input_into_arm_states(sensory_inputs=sensory_input,
                                                          states=states)
        embed_vectorized_apply = jax.jit(jax.vmap(self._model_embed.apply))
        output_vectorized_apply = jax.jit(jax.vmap(self._model_output.apply))

        # policy_params for every arm identical for every arm
        policy_params_embed = self._model_embed.policy_params 
        policy_params_output = self._model_output.policy_params
        # If HebbianController: policy params are learning rules and biases
        # If NNController: policy params are synapse strengths and biases

        # embed
        for i in range(len(self._config["morphology"]["arm_setup"])):
            sensory_input_embed = jnp.array(states[f"arm_{i}_embed"]["inputs"][:,-1,:])

            if self._config["controller"]["hebbian"] == True:
                synapse_strengths_embed = self.get_synapse_strengths_from_arm_states(states=states, arm_ind=i, embed_or_output="embed")
                neuron_activities_embed = self.get_neuron_activities_from_arm_states(states=states, arm_ind=i, embed_or_output="embed")
                actuator_output_embed, synapse_strengths_embed, neuron_activities_embed = embed_vectorized_apply(
                                                                    sensory_input=sensory_input_embed,
                                                                    synapse_strengths=synapse_strengths_embed,
                                                                    learning_rules=policy_params_embed,
                                                                    neuron_activities=neuron_activities_embed
                                                                )
            elif self._config["controller"]["hebbian"] == False:
                synapse_strengths_embed = policy_params_embed
                actuator_output_embed, neuron_activities_embed = embed_vectorized_apply(
                                                                    sensory_input=sensory_input_embed,
                                                                    synapse_strengths=synapse_strengths_embed                    
                                                                )
            
            states['central_reservoir'][f"arm_{i}"] = jnp.concatenate([states['central_reservoir'][f"arm_{i}"], jnp.expand_dims(actuator_output_embed, axis = 1)],
                                                                  axis = 1)
            

            for layer_ind in range(len(self._embed_layers)-1):
                states[f"arm_{i}_embed"][f"layers_{layer_ind}"]["output"] = jnp.concatenate([states[f"arm_{i}_embed"][f"layers_{layer_ind}"]["output"],
                                                                                             jnp.expand_dims(neuron_activities_embed[layer_ind+1], axis = 1)],
                                                                                             axis = 1)
                states[f"arm_{i}_embed"][f"layers_{layer_ind}"]["kernel"] = jnp.concatenate([states[f"arm_{i}_embed"][f"layers_{layer_ind}"]["kernel"],
                                                                                             jnp.expand_dims(synapse_strengths_embed["params"][f"layers_{layer_ind}"]["kernel"], axis = 1)],
                                                                                             axis = 1)
                states[f"arm_{i}_embed"][f"layers_{layer_ind}"]["bias"] = jnp.concatenate([states[f"arm_{i}_embed"][f"layers_{layer_ind}"]["bias"],
                                                                                jnp.expand_dims(synapse_strengths_embed["params"][f"layers_{layer_ind}"]["bias"], axis = 1)],
                                                                                axis = 1)


        # output
        for i in range(len(self._config["morphology"]["arm_setup"])):
            sensory_input_output =  self.permutation_central_reservoir(states=states, arm_index=i) # sensory input to output layer

            states[f"arm_{i}_output"]["inputs"] = jnp.concatenate([states[f"arm_{i}_output"]["inputs"],
                                                                             sensory_input_output],
                                                                             axis = 1)
            
            sensory_input_output = sensory_input_output[:,-1,:]


            if self._config["controller"]["hebbian"] == True:
                synapse_strengths_output = self.get_synapse_strengths_from_arm_states(states=states, arm_ind=i, embed_or_output="output")
                neuron_activities_output = self.get_neuron_activities_from_arm_states(states=states, arm_ind=i, embed_or_output="output")
                actuator_output_output, synapse_strengths_output, neuron_activities_output = output_vectorized_apply(
                                                                    sensory_input=sensory_input_output,
                                                                    synapse_strengths=synapse_strengths_output,
                                                                    learning_rules=policy_params_output,
                                                                    neuron_activities=neuron_activities_output
                                                                )
                
            elif self._config["controller"]["hebbian"] == False:
                synapse_strengths_output = policy_params_output
                actuator_output_output, neuron_activities_output = output_vectorized_apply(
                                                                    sensory_input=sensory_input_output,
                                                                    synapse_strengths=synapse_strengths_output                    
                                                                )

            for layer_ind in range(len(self._output_layers)-1):
                states[f"arm_{i}_output"][f"layers_{layer_ind}"]["output"] = jnp.concatenate([states[f"arm_{i}_output"][f"layers_{layer_ind}"]["output"],
                                                                                             jnp.expand_dims(neuron_activities_output[layer_ind+1], axis = 1)],
                                                                                             axis = 1)
                states[f"arm_{i}_output"][f"layers_{layer_ind}"]["kernel"] = jnp.concatenate([states[f"arm_{i}_output"][f"layers_{layer_ind}"]["kernel"],
                                                                                             jnp.expand_dims(synapse_strengths_output["params"][f"layers_{layer_ind}"]["kernel"], axis = 1)],
                                                                                             axis = 1)
                states[f"arm_{i}_output"][f"layers_{layer_ind}"]["bias"] = jnp.concatenate([states[f"arm_{i}_output"][f"layers_{layer_ind}"]["bias"],
                                                                                jnp.expand_dims(synapse_strengths_output["params"][f"layers_{layer_ind}"]["bias"], axis = 1)],
                                                                                axis = 1)
        
        actuation_signal_brittle_star_agent = self.get_actuation_signal_brittle_star_agent_from_arm_states(states=states)

        return actuation_signal_brittle_star_agent, states
    


    def store_sensory_input_into_arm_states(
            self,
            sensory_inputs: chex.Array,
            states: dict
    ) -> dict:
        """
        Pure function, no side effects

        Sensory inputs are a 2D array with dim = (#popsize, #sensors).
        Only the sensors from the sensor selection are included.
        """
        if self._config["controller"]["decentralized"]["explicit_body_sensors"]:
            raise NotImplementedError("sensory input not yet incorporated into the central reservoirs body sensors")

        sensory_inputs_expanded = jnp.expand_dims(sensory_inputs, axis = 1) # reflects the timestep dim. Should be added to allow concatenations later.

        sensor_selection = self._config["environment"]["sensor_selection"]
        num_arms = len(self._config["morphology"]["arm_setup"])
        num_segm_per_arm = self._config["morphology"]["arm_setup"][0]
        check_sensor_selection_order(sensor_selection)
        
        segm_sensors_2 = ['joint_position', 'joint_velocity', 'joint_actuator_force'] # every segment has 2 sensors: ip and oop
        segm_sensors_1 = ['segment_contact', 'segment_light_intake'] # every segment has 1 sensor
        body_sensors_3D = ['disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity'] # sensors are related to body
        body_sensors_2D = ['unit_xy_direction_to_target']
        body_sensors_1D = ['xy_distance_to_target']

        
        # make empty array to store the new sensory input per arm
        tmp_dict = {}
        for i in range(num_arms):
            tmp_dict[f"arm_{i}"] = []


        # Extract new sensory input and distribute in correct order across arms
        for sensor in sensor_selection:
            if sensor in segm_sensors_2:
                dim = 2*num_segm_per_arm
                for i in range(num_arms):
                    tmp_dict[f"arm_{i}"].append(sensory_inputs_expanded[:,:,:dim])                    
                    sensory_inputs_expanded = sensory_inputs_expanded[:,:,dim:]

            if sensor in segm_sensors_1:
                dim = num_segm_per_arm
                for i in range(num_arms):
                    tmp_dict[f"arm_{i}"].append(sensory_inputs_expanded[:,:,:dim])                    
                    sensory_inputs_expanded = sensory_inputs_expanded[:,:,dim:]

            if sensor in body_sensors_3D:
                dim = 3
                sensory_inputs_expanded = sensory_inputs_expanded[:,:,dim:]
                # you could add stuff here to central_reservoir later if needed
            
            if sensor in body_sensors_2D and not sensor=='unit_xy_direction_to_target':
                dim = 2
                sensory_inputs_expanded = sensory_inputs_expanded[:,:,dim:]
                # you could add stuff here to central_reservoir later if needed
            
            elif sensor in body_sensors_2D and sensor=='unit_xy_direction_to_target' and self._config["environment"]["reward_type"] == 'target':
                dim = 2
                arm_target_dir_projections = calculate_arm_target_allignment_factors(sensory_inputs_expanded[:,-1,:dim])
                arm_target_dir_projections_expanded = jnp.expand_dims(arm_target_dir_projections, axis = 1)
                for i in range(num_arms):
                    tmp_dict[f"arm_{i}"].append(jnp.expand_dims(arm_target_dir_projections_expanded[:,:,i], axis=-1))
                sensory_inputs_expanded = sensory_inputs_expanded[:,:,dim:]
                # you could add stuff here to central_reservoir later if needed
            

            if sensor in body_sensors_1D:
                dim = 1
                sensory_inputs_expanded = sensory_inputs_expanded[:,:,dim:]
                # you could add stuff here to central_reservoir later if needed

        assert sensory_inputs_expanded.shape[-1] == 0, "Implementation incorrect: the array should be empty after all the sensors have been considered."

        
        # Add this sensory information per arm to the states dict
        for i in range(num_arms):
            tmp_dict[f"arm_{i}"] = jnp.concatenate(tmp_dict[f"arm_{i}"], axis=-1)
            states[f"arm_{i}_embed"]["inputs"] = jnp.concatenate([self.states[f"arm_{i}_embed"]["inputs"],\
                                                                tmp_dict[f"arm_{i}"]],
                                                                axis=1)

        
        return states



    def permutation_central_reservoir(
            self,
            states: dict,
            arm_index
    ) -> chex.Array:
        """
        input:
        - arm_index, prescribing to what extent the array should be permutated
        output:
        - array with dimension (popsize, dim of central reservoir), so no dimension on axis 1 with the timesteps

        No side effects
        """
        
        permutation_central_reservoir = []
        num_arms = len(self._config["morphology"]["arm_setup"])

        for i in range(num_arms):
            arm_number = (i+arm_index) % num_arms
            permutation_central_reservoir.append(states['central_reservoir'][f"arm_{arm_number}"][:,-1,:])
            

        if self._config["controller"]["decentralized"]["explicit_body_sensors"]:
            permutation_central_reservoir.append(states['central_reservoir']["body_sensors"][:,-1,:])


        permutation_central_reservoir = jnp.concatenate(permutation_central_reservoir, axis=-1)
        permutation_central_reservoir = jnp.expand_dims(permutation_central_reservoir, axis = 1)
        return permutation_central_reservoir
    

    def get_neuron_activities_from_arm_states(
            self,
            states: dict,
            arm_ind: int,
            embed_or_output: Literal["embed", "output"]
    ) -> list[chex.Array]:
        """
        neuron activities are a list with the node activities of all the neurons, so
        [input_nodes, activities_hidden_layer_1, activities_hiden_layer_2, ..., activities_outputs_nodes]

        no side effects
        """
        neuron_activities = []
        neuron_activities.append(states[f"arm_{arm_ind}_{embed_or_output}"]["inputs"][:,-1,:])
        if embed_or_output == "embed":
            for p in range(len(self._embed_layers)-1):
                neuron_activities.append(states[f"arm_{arm_ind}_{embed_or_output}"][f"layers_{p}"]["output"][:,-1,:])
        elif embed_or_output == "output":
            for p in range(len(self._output_layers)-1):
                neuron_activities.append(states[f"arm_{arm_ind}_{embed_or_output}"][f"layers_{p}"]["output"][:,-1,:])

        return neuron_activities


    def get_synapse_strengths_from_arm_states(
            self,
            states: dict,
            arm_ind: int,
            embed_or_output: Literal["embed", "output"]
    ) -> dict:
        """
        returns dict with the following structure:
        {params:
            {layers_0:
                {kernel: array(popsize, 2D kernel dim)
                 bias:   array(popsize, 1D bias dim)
                },
             layers_1:
                ...,
             ...
            }
        }

        No side effects!
        """

        synapse_strengths_arm = {}
        synapse_strengths_arm["params"] = {}
        if embed_or_output == "embed":
            for p in range(len(self._embed_layers)-1):
                synapse_strengths_arm["params"][f"layers_{p}"] = {}
                synapse_strengths_arm["params"][f"layers_{p}"]["kernel"] = states[f"arm_{arm_ind}_{embed_or_output}"][f"layers_{p}"]["kernel"][:,-1,:,:]
                synapse_strengths_arm["params"][f"layers_{p}"]["bias"] = states[f"arm_{arm_ind}_{embed_or_output}"][f"layers_{p}"]["bias"][:,-1,:]

        if embed_or_output == "output":
            for p in range(len(self._output_layers)-1):
                synapse_strengths_arm["params"][f"layers_{p}"] = {}
                synapse_strengths_arm["params"][f"layers_{p}"]["kernel"] = states[f"arm_{arm_ind}_{embed_or_output}"][f"layers_{p}"]["kernel"][:,-1,:,:]
                synapse_strengths_arm["params"][f"layers_{p}"]["bias"] = states[f"arm_{arm_ind}_{embed_or_output}"][f"layers_{p}"]["bias"][:,-1,:]

        return synapse_strengths_arm


    def get_actuation_signal_brittle_star_agent_from_arm_states(
        self,
        states: dict
    ) -> chex.Array:
        """
        Extract actuation signal from arm states (concatenates output of output network of every arm)

        No side effects
        """
        actuation_signal = []
        num_layers = len(self._output_layers)-1
        for i in range(len(self._config["morphology"]["arm_setup"])):
            arm_actuation_signal = states[f"arm_{i}_output"][f"layers_{num_layers-1}"]["output"][:,-1,:]
            actuation_signal.append(arm_actuation_signal)
        
        actuation_signal = jnp.concatenate(actuation_signal, axis = -1)
        return actuation_signal




        
    

    




            
    

    




