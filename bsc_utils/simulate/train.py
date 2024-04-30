

from bsc_utils.BrittleStarEnv import EnvContainer



class Trainer(EnvContainer):
    """
    A Trainer is an EnvContainer, as it contains all the necessary environments used during training.
    On top of these methods and attributes, it allows all functionalities to actually perform the training.
    """
    def __init__(self,
                 config: dict
                 ):
        super().__init__(config)


# opportunities to add more classes?