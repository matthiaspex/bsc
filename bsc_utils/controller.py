# controller
import chex
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
        -----
        
        """
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
        return x
    


class NNController():
    def __init__(
            self,
            env_container: EnvContainer # hence also classes inheriting from EnvContainer are fine
            ):
        self.config = env_container.config
        self.env_container = env_container


    def update_model(
            self,
    ):
        assert self.env_container.env, "Model must be build based on undamaged configurations, but no non-damaged env was instatiated"
        nn_input_dim, nn_output_dim = self.env_container.get_observation_action_space_info()
        self.layers = [nn_input_dim] + self.config["controller"]["hidden_layers"] + [nn_output_dim]
        _features = tuple(self.layers[1:])
        self.model = ExplicitMLP(features = _features, joint_control = self.config["morphology"]["joint_control"])
    
    def get_policy_params_example(
            self
    ) -> dict:
        """
        models are based on flax, so policy_params are characterized by pytrees
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

        

