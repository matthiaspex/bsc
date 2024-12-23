from bsc_utils.controller.base import ExplicitMLP, NNController
from bsc_utils.controller.hebbian import HebbianController
from bsc_utils.simulate.analyze import Simulator
from typing import List, Union
import jax

class Consolidator(Simulator):
    def __init__(self,
                 config:dict
    ):
        super().__init__(config)

    
    def build_mimic_controller(
            self,
            hidden_layers: List,
            means: List = [-3,3],
            stds: List = [1,1],
            trunc_mins: List = [-6,0],
            trunc_maxs: List = [0,6]
    ):
        """
        Input:
        - layers: [input dim, hidden dims, output dim]
        """
        input_dim, output_dim = self.get_observation_action_space_info()
        self.mimic_layers = [input_dim] + self.config["controller"]["hidden_layers"] + [output_dim]
        self.mimic_controller = ExplicitMLP(features = self.mimic_layers[1:], joint_control=self.config["morphology"]["joint_control"])
        mimic_controller_params = self.mimic_controller.init(jax.random.PRNGKey(0),
                                                             jax.random.uniform(
                                                                 jax.random.PRNGKey(1),
                                                                 (self.mimic_layers[0])
                                                            )
                                                        )
        