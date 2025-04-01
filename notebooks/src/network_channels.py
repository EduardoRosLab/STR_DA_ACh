import numpy as np

from network_base import Network_Base
from channel import Channel


class Network_Channels(Network_Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.non_output_channels = 0

        self.d1_learning_rate = 1e-3
        self.d2_learning_rate = 1e-2

        self.with_ach = True

        # Lateral connectivity patterns
        self.d1d1_inter = 3.0 #0.0 #3.0
        self.d1d2_inter = 0.0 #0.0
        self.d2d1_inter = 0.0 #0.0
        self.d2d2_inter = 3.0 #0.0 #3.0

        self.max_weight = 10.0  #0.2,
        self.max_init_weight_d1 = 8.0  #0.2,
        self.min_init_weight_d1 = 0.2 
        self.max_init_weight_d2 = 2.0  #0.05,
        self.min_init_weight_d2 = 0.05  #0.03,


    def _init_vars(self):
        self.n_total_channels = self.n_actions
        super()._init_vars()
    

    def _init_network(self):
        self._init_layers()
        self._init_auxiliar_vars()
        self._init_channels()
        self._init_lateral_connections()
        self.sim.Initialize()


    def _init_channels(self):
        self.channels = []
        for action_neuron in self.action_layer:
            # Create the channel
            channel = Channel(network=self, action_neuron=action_neuron, **self.kwargs)
            self.channels.append(channel)

            # Connect the channel to the action neuron
            _ = self.sim.AddSynapticLayer(
                channel.msn_d1_layer, [action_neuron]*len(channel.msn_d1_layer), 
                {**self.excitatory_synapse_params, 'weight': 0.133})
            _ = self.sim.AddSynapticLayer(
                channel.msn_d2_layer, [action_neuron]*len(channel.msn_d2_layer),
                {**self.inhibitory_synapse_params, 'weight': 0.133})
        self.channels = np.array(self.channels)
        
    
    def _init_action_neuron_lateral_connections(self):
        for action_neuron in self.action_layer:
            tgt = [o for o in self.action_layer if o!=action_neuron]
            _ = self.sim.AddSynapticLayer(
                [action_neuron]*len(tgt), tgt, 
                {**self.inhibitory_synapse_params, 'weight': 10.0}) 


    def _init_channel_lateral_connections(self):
        for cha_src in self.channels:
            for cha_tgt in self.channels:
                if cha_src==cha_tgt: continue

                connections = [
                    (cha_src.msn_d1_layer, cha_tgt.msn_d1_layer, self.d1d1_inter),
                    (cha_src.msn_d1_layer, cha_tgt.msn_d2_layer, self.d1d2_inter),
                    (cha_src.msn_d2_layer, cha_tgt.msn_d1_layer, self.d2d1_inter),
                    (cha_src.msn_d2_layer, cha_tgt.msn_d2_layer, self.d2d2_inter),
                ]
                for (source_layer, target_layer, connection_weight) in connections:
                    sources, targets, weights = [], [], []
                    for ni in source_layer:
                        for no in target_layer:
                            if ni==no or connection_weight==0.0: continue
                            weights.append(connection_weight)
                            sources.append(ni)
                            targets.append(no)
                    if len(sources)>0 and len(targets)>0:
                        _ = self.sim.AddSynapticLayer(
                            sources, targets,
                            {**self.inhibitory_synapse_params, 'weight': weights})


    def _init_lateral_connections(self):
        self._init_action_neuron_lateral_connections()
        self._init_channel_lateral_connections()