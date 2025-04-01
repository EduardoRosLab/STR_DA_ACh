from task_state_population import Task_StatePopulation, Task_StatePopulationRelearning
from task_proanti import Task_ProAnti
from network_channels import Network_Channels
from misc import Params_Base


class Experiment_Base:
    def __init__(self, **kwargs):
        
        # Initialize the base classes
        for parent in self.__class__.__bases__:
            if parent is Experiment_Base: continue
            parent.__init__(self, **kwargs)

        # Update the attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.kwargs = kwargs.copy() if type(kwargs) is dict else None

        # Create variables in the base classes
        for parent in self.__class__.__bases__:
            if hasattr(parent, '_init_vars'):
                parent._init_vars(self)
        
        # Create the network in the base classes
        for parent in self.__class__.__bases__:
            if hasattr(parent, '_init_network'):
                parent._init_network(self)


class Experiment_StatePopulationsWithChannels(
    Experiment_Base, Params_Base, Task_StatePopulation, Network_Channels):
    pass

class Experiment_StatePopulationWithChannelsReLearning(
    Experiment_Base, Params_Base, Task_StatePopulationRelearning, Network_Channels
): pass

class Experiment_ProAntiWithChannels(
    Experiment_Base, Params_Base, Task_ProAnti, Network_Channels):
    pass
