# controller
import chex
from typing import Union, Type

import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence
from evosax import ParameterReshaper

from bsc_utils.BrittleStarEnv import EnvContainer



# build NN architecture
class ExplicitMLP(nn.Module):
    features: Sequence[int]
    joint_control: str 
    # act_hidden: Callable = nn.tanh,
    # act_output: Callable = nn.tanh

    """
    features: number of outputs (# nodes) for each layer. The number of inputs of the first layer defined by the call function.
    The number of inputs of the hidden layers and output layer defined by the number of outputs of the previous layer.
    act_hidden: activation function applied to hidden layers: popular is nn.tanh or nn.relu
    act_output: activation function applied to output layer: popular is nn.tanh or nn.sigmoid
    joint_control: should be either 'position' or 'torque', depending on what control strategy the morphology was initialised with
    """

    def setup(
        self
    ):
        """
        Fully connected neural network, characterised by a pytree (dict containing dict with params and biases)
        Features represents the number of outputs of the Dense layer
        inputs based on the presented input later on
        after presenting input: kernel can be generated
        """
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(
        self,
        inputs,
        act_hidden: Callable = nn.tanh,
        act_output: Callable = nn.tanh
    ):
        """
        Returning the output of a layer for a given input.
        Don't directly call an instance of ExplicitMLP --> this method is called in the apply method.
        
        output:
        - x: output array of neural network
        - neuron_activities: neuron activities of the neural network, including the input layer.
            e.g. neural network with 2 hidden layers has 4 layers of node activities:
            [input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes]
        """
        neuron_activities = [inputs]
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = act_hidden(x)
            else:
                assert self.joint_control in ['position', 'torque'], "joint_control should be either 'position' or 'torque'"
                if self.joint_control == 'position':
                    x = 30*jnp.pi/180 * act_output(x) # the action space range for positions is -0.5236..0.5236
                elif self.joint_control == 'control':
                    x = act_output(x) # the action space range for torques is -1..1
            neuron_activities.append(x)
        return x, neuron_activities
    


class NNController():
    def __init__(
            self,
            env_container: EnvContainer # hence also classes inheriting from EnvContainer are fine, like Simulator
            # A similator is an EnvContainer because it contains everything necessary for environment
            ):
        self.config = env_container.config
        self.env_container = env_container


    def update_model(
            self,
            model_class: Union[Type[ExplicitMLP], None] = ExplicitMLP, # provide ExplicitMLP as model class, not an instance of it.
            layer_architecture: Union[list, None] = None
    ):
        """
        Model updated based on env_container (input and output dimensions) and the configuration file (hidden layers)
        Inputs:
        - model_class: model must be supported by evosax, built based on Flax. Stuff like Parameter reshaper must still work
        - layer_architecture: array with [#input nodes, #hidden nodes, #output nodes]
        e.g. 50 sensors, 2x128 hidden nodes and 20 actuators would be [50, 128, 128, 20]
        -> By default, the layer architecture is based on the number of sensors and actuators,
        and the number of hidden layers of a centralized controller.
        When layer_architecture is provided, this is no longer the case.
        ---
        Generates a self.model attribute which is by default of the type ExplicitMLP
        """
        assert self.env_container.env, "Model must be built based on undamaged configurations, but no non-damaged env was instantiated"
        if layer_architecture == None:
            nn_input_dim, nn_output_dim = self.env_container.get_observation_action_space_info()
            self.layers = [nn_input_dim] + self.config["controller"]["hidden_layers"] + [nn_output_dim]
        else:
            self.layers = layer_architecture

        if model_class == ExplicitMLP:
            _features = tuple(self.layers[1:])
            self.model = ExplicitMLP(features = _features, joint_control = self.config["morphology"]["joint_control"])
        else:
            raise TypeError("No valid model class was provided")
    
    def get_policy_params_example(
            self
    ) -> dict:
        """
        models are based on flax, so policy_params are characterized by pytrees
        Return: dict: pytree of Layers: kernels, biases --> only parallel pytree, no popsize or anything
        Useful as hidden method for update_parameter_reshaper
        """
        assert self.model, "No model has been instantiated yet. Use method update_model"
        policy_params_example = self.model.init(
                                    jax.random.PRNGKey(0),          # model initialiser PRNGKey
                                    jax.random.uniform(             # generate a random input vector
                                        jax.random.PRNGKey(2024),   # seed for random input
                                        (self.layers[0],)           # (iterable) dimension of the input
                                    )
                                )
        return policy_params_example


    def reset_neuron_activities(
            self,
            parallel_dim: int
    ):
        """
        Generate a sequence of neuron activities.
        Note: when vmap: sequence is treated as a tree, elements in the sequence (being arrays) are leafs
        When vmapped, it is by default the first dim of the leafs that is considered as mappable dimension
        """
        neuron_activities = []
        for nodes in self.layers:
            neuron_activities.append(jnp.zeros((parallel_dim, nodes)))
        self.neuron_activities = neuron_activities

    def get_neuron_activities(
            self,
    ) -> Sequence[chex.Array]:
        return self.neuron_activities
    
    def update_parameter_reshaper(self):
        """
        Instantiates attribute self.parameter_reshaper (as an object of evosax.ParameterReshaper class)
        """
        assert self.model, "No model has been instantiated yet. Use method update_model"
        policy_params_example = self.get_policy_params_example()
        self.parameter_reshaper = ParameterReshaper(policy_params_example)
    
    def get_parameter_reshaper(self) -> ParameterReshaper:
        """
        Explicitly define this function to make clear that it is possible to query the parameter_reshaper
        """
        assert self.parameter_reshaper, "first instantiate the self.parameter_reshaper attribute by calling update_parameter_reshaper method"
        return self.parameter_reshaper
    
    def update_policy_params(
            self,
            policy_params: Union[dict, chex.Array]
    ):
        """
        Stores a dict, but can take evosax array as flat input
        """
        if isinstance(policy_params, dict):
            self.policy_params = policy_params
        if isinstance(policy_params, chex.Array):
            self.policy_params = self.parameter_reshaper.reshape(policy_params)

    def get_policy_params(
            self,
    ) -> dict:
        assert self.policy_params, "first provide policy params using the method update_policy_params"
        return self.policy_params

    def update_synapse_strengths(
            self,
            synapse_strengths: Union[dict, chex.Array] # takes flat inputs from evosax format or the reshaped formats
    ):
        assert self.parameter_reshaper, "No parameter_reshaper has been instantiated yet. Use method update_parameter_reshaper"
        if isinstance(synapse_strengths, dict): # pytree format
            # check whether number of parameters matches the ParameterReshaper
            required_num_params = self.parameter_reshaper.total_params

            sizes = []
            jax.tree_util.tree_map(lambda x: sizes.append(jnp.size(x[0])), synapse_strengths) # select only 1 parallel evolution parameter: x[0]
            # jnp.size checks the number of parameters in the kernels
            actual_num_params = sum(sizes)


            assert actual_num_params == required_num_params, \
                """shape of the provided learning_rules params doesn't match the one required according to ParamReshaper
                   Or: problem with vmapping: learning rules is a 1D array instead of 2D array with shape (# parellel simulations, # policy params)"""
            self.synapse_strengths = synapse_strengths
            
        elif isinstance(synapse_strengths, chex.Array): # flat evosax format with dim (popsize, parameter.total_params)
            self.synapse_strengths = self.parameter_reshaper.reshape(synapse_strengths)


    def apply(
            self,
            sensory_input: chex.Array,
            synapse_strengths: dict # just provide self.synapse_strengths, can't be in here because of vmapping power
    ):
        """
        Still needs to be vectorized when it is being applied to pytrees of synaptic strengths with first dim = popsize
        For example: synapse_strenghts are dict with leave shapes:
        {Params: Layer0: Kernel: (1,125,128)} -> jax.vmappable, even though the axis to map has len 1
        {Params: Layer0: Kernel: (6912,125,128)} -> jax.vmap will map over 6912 parallel candidate policy params

        synapse_strengths must already be a dict
        """
        assert self.model, "First provide a model by using the update_model method"

        actuator_output, neuron_activities = self.model.apply(synapse_strengths, sensory_input)

        return actuator_output, neuron_activities
    


        

