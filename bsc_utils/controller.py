# controller
import numpy as np
import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
from typing import Any, Callable, Sequence, Union, List



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
    


class nn_controller():
    def __init__(
            self,
            cfg: dict,
            output_dim: int
            ):
        self.cfg = cfg
        self.output_dim = output_dim


    def model_from_config(
            self,
    ):
        self.features = tuple(self.cfg["controller"]["hidden_layers"] + [self.output_dim])
        self.model = ExplicitMLP(features = self.features, joint_control = self.cfg["morphology"]["joint_control"])
        

