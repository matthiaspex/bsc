from typing import Union, Sequence
import chex
import copy

import jax
from jax import numpy as jnp
from evosax import ParameterReshaper

from bsc_utils.controller.base import NNController
from bsc_utils.BrittleStarEnv import EnvContainer

class HebbianController(NNController):
    def __init__(
            self,
            env_container: EnvContainer
    ):
        super().__init__(env_container)

    def get_policy_params_example(self) -> dict:
        """
        For a Hebbian controller, the policy parameters are the learning rules, not the synapses
        First need to have a model defined.
        """
        synapse_strenghts_example = super().get_policy_params_example() # get synapse strengths tree example from basic static MLP
        policy_params_empty = self._empty_learning_rule_tree_from_synapse_strengths(synapse_strenghts_example)
        return policy_params_empty
    

    def _empty_learning_rule_tree_from_synapse_strengths(
            self,
            synapse_strengths: dict
    ) -> dict:
        """
        Apply on pytrees containtining 2D kernals and 1D bias arrays
        Adds a learning rule of n parameters to each kernel in an additional dimension
        Only applies this to kernels, not to bias arrays
        each kernel of shape (i,j) becomes a kernel of shape (i,j,n)
        each bias of shape (m,) remains bias of shape (m,)
        RETURNED ARRAY CONTAINS ZEROES (only shapes of learning rules)
        """
        learning_rules_empty = jax.tree_util.tree_map(lambda x: self._add_learning_rule_dim(x,n=5), synapse_strengths)
        return learning_rules_empty

    def _add_learning_rule_dim(
            self,
            x,
            n=5
    ):
        """
        The callable to be used by method _empty_learning_rule_tree_from_synapse_strenghts
        For 2D kernels matrices, it appends a fifth dimension with n elements (the learning rule parameters)
        For 1D bias arrays, it does nothing
        It returns a jnp.zeros array of the provided size.
        """
        if len(x.shape) == 2:
            new_shape = tuple(list(x.shape)+[n])
            return jnp.zeros(new_shape)
        else: # for biases
            return jnp.zeros_like(x)

    def reset_synapse_strengths_unif(
            self,
            rng: chex.PRNGKey,
            parallel_dim: int, # either popsize during training or just 1 or multiple parallel environments during evaluation afterwards
            minval = -0.1,
            maxval = 0.1            
    ):
        """
        Based on Pedersen (2021) who initialize synapse strengths uniformly at beginning of episode between -0.1 and 0.1
        """
        ss_example = super().get_policy_params_example() # gives a pytree with synapse strengths
        ss_parameter_reshaper = ParameterReshaper(ss_example)
        print("num_params from HebbianController class method reset_synapse_strengths_unif")
        ss_num_params = ss_parameter_reshaper.total_params
        synapse_strengths_init_flat = jax.random.uniform(rng, shape=(parallel_dim, ss_num_params), minval = minval, maxval = maxval)

        self.synapse_strengths = ss_parameter_reshaper.reshape(synapse_strengths_init_flat)

    def get_synapse_strengths(
            self
    ) -> dict:
        return self.synapse_strengths


    def update_parameter_reshaper(self):
        assert self.model, "No model has been instantiated yet. Use method update_model"
        policy_params_empty = self.get_policy_params_example() # gives empty pytree with learning rules
        self.parameter_reshaper = ParameterReshaper(policy_params_empty)

    
    
    def get_parameter_reshaper(self) -> ParameterReshaper:
        assert self.parameter_reshaper, "first instantiate the self.parameter_reshaper attribute by calling update_parameter_reshaper method"
        return self.parameter_reshaper
    

    def update_learning_rules(
            self,
            learning_rules: Union[dict, chex.Array] # can be pytree with learning rules or flat array from evosax ask
    ):
        """
        Simply takes learning_rules as dict or flat evosax array
        Dict input: check whether number of parameters compatible with current ParameterReshaper
        Evosax Array input with dim (popsize, parameter.total_params): apply the ParameterReshaper to get dict
        Overwrites self.learning_rules to be a dict (pytree)
        """
        assert self.parameter_reshaper, "no Parameter_reshaper has been instantiated yet. Use method update_parameter_reshaper"
        if isinstance(learning_rules, dict): # pytree format: check whether the number of parameters is correct
            required_num_params = self.parameter_reshaper.total_params

            sizes = []
            jax.tree_util.tree_map(lambda x: sizes.append(jnp.size(x[0])), learning_rules) # select only 1 parallel evolution parameter: x[0]
            actual_num_params = sum(sizes)
            
            assert actual_num_params == required_num_params, \
                """shape of the provided learning_rules params doesn't match the one required according to ParamReshaper
                   Or: problem with vmapping: learning rules is a 1D array instead of 2D array with shape (# parellel simulations, # policy params)"""
            self.learning_rules = learning_rules

        elif isinstance(learning_rules, chex.Array): # flat evosax format with dim (popsize, parameter.total_params)
            self.learning_rules = self.parameter_reshaper.reshape(learning_rules)


    def apply(
            self,
            sensory_input: chex.Array,
            synapse_strengths: dict, # provide self.synapse_strengths, can't be in here because of vmapping power
            learning_rules: dict, # provide self.learning_rules, can't be in here because of vmapping power
            neuron_activities: Sequence[chex.Array] # provide self.neuron_activities
    ):
        """
        Allow for VMAP:
        - every input inputted here should have the vmap input dim, but ...
        - ...every function within should not rely on input dim
        --> it can function without an axis to map over, but it can be vmapped to take inputs with an additional direction to map over

        synapse strengths: the initial synapse_strengths to add increment to.
        
        Provide sensory input to pass through the model
        Provide the learning rules as a flat array or as a pytree (dict)
        Pytree kernel leave shape should be: (popsize, input layer dim, output layer dim, 5) (with 5 the dimension of the learning rule)
        This function updates the self.synapse_strengths and passes those on to the super apply method.

        return:
        - actuator_output: necessary to take a step in the simulation environment
        - synapse_strengths: dict, necessary to update in next apply step
        - neuron_activities: necessary for next apply of a plastic neural network
        IMPORTANT: DON'T ALLOW SIDE EFFECTS IN FUNCTION THAT WILL HAVE TO BE JITTED
        --> don't overwrite attributes like self.synapse_strengths or neuron_activities

       - neuron_activities: neuron activities of the neural network, including the input layer.
        e.g. neural network with 2 hidden layers has 4 layers of node activities:
        [input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes]
        """

        # implement all functionalities that update the synapse strengths based on the learning rules
        synapse_strengths = self._update_synapse_strengths(synapse_strengths,
                                                           learning_rules,
                                                           neuron_activities)

        # pass on the new synapse strengths to the super.apply() method
        actuator_output, neuron_activities = super().apply(sensory_input=sensory_input, synapse_strengths=synapse_strengths)

        return actuator_output, synapse_strengths, neuron_activities
    

    vectorized_apply = jax.jit(jax.vmap(apply))
    # this function is used for instance by the DecentralizedControlller class

    def _update_synapse_strengths(
            self,
            synapse_strengths_input: dict,
            learning_rules: dict, # pytree
            neuron_activities: Sequence[chex.Array] # jax array
    ) -> dict:
        """
        This function is HebbianController specific and is meant to update the synaptic strengths to apply in the MLP as defined in the NNController superclass.
        """
        # problem: for some reason adjusting the dicts in this function causes the global value to be affected. For security, use copy.deepcopy
        synapse_strengths = copy.deepcopy(synapse_strengths_input)

        num_layers = len(learning_rules["params"].keys())
        for p in range(num_layers):
            lr_kernel = learning_rules["params"][f"layers_{p}"]["kernel"]
            input_nodes = neuron_activities[p]
            output_nodes = neuron_activities[p+1]

            ss_incr_kernel = self._apply_learning_rule(lr_kernel, input_nodes, output_nodes)

            synapse_strengths["params"][f"layers_{p}"]["kernel"] += ss_incr_kernel
            synapse_strengths["params"][f"layers_{p}"]["bias"] = learning_rules["params"][f"layers_{p}"]["bias"]

        return synapse_strengths


    def _apply_learning_rule(self,
                             lr_kernel,
                             input_nodes,
                             output_nodes,
                             learning_rule: str = "ABCD"
                             ):
        
        """ 
        Input: 
        - learning rule kernel: dims (popsize, input_layer_dim, output_layer_dim, lr_dim = 5)
        - learning_rule: only "ABCD" implemented so far.
        Output: synaptic strength increment kernel: dims (popsize, input_layer_dim, output_layer_dim)
        This function is vmapable and jittable
        """
        assert lr_kernel.shape[-1] == 5, "Learning rule requires 5 parameters or needs to be updated so it is compatible with different number of parameters"
        in_dim = len(input_nodes)
        out_dim = len(output_nodes)
        inp = jnp.transpose(jnp.tile(input_nodes, (out_dim, 1))) # Generates (in_dim, out_dim) dimension, but constant along axis = 1 (output dimesnion)
        outp = jnp.tile(output_nodes, (in_dim, 1)) # Generates (in_dim, out_dim) dimension, but constant along axis = 0 (input dimesnion)

        # kernel content: [alpha, A, B, C, D] --> Dw_ij = alpha_ij * (A_ij*o_i*o_j + B_ij * o_i + C_ij * o_j + D_ij)
        alpha = lr_kernel[:,:,0]
        A = lr_kernel[:,:,1]
        B = lr_kernel[:,:,2]
        C = lr_kernel[:,:,3]
        D = lr_kernel[:,:,4]

        if learning_rule == "ABCD":
            pass
        elif learning_rule == "AD":
            raise NotImplementedError
            # you could do something like B *= 0 and then the below will still apply.

        ss_incr_kernel = alpha * (A*inp*outp + B*inp + C*outp + D)

        return ss_incr_kernel # synaptic strength increment kernel
    

    

    

