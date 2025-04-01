import numpy as np
from edlut_python_2_rules import simulation_wrapper as pyedlut


class Network_Base:
    def __init__(self, **kwargs):
        pass

    def _init_vars(self):
        if not getattr(self, 'sim', None): self.sim = pyedlut.PySimulation_API()
        self._init_edlut_rules()

    def _init_network(self):
        self._init_layers()
        self._init_auxiliar_vars()

    def _init_auxiliar_vars(self):
        self.ot, self.oi = [], []
        self.channel_d1_weights  = [[] for _ in range(self.n_total_channels)]
        self.channel_d2_weights  = [[] for _ in range(self.n_total_channels)]
        self.channel_fsi_weights = [[] for _ in range(self.n_total_channels)]
        self.trial_times = []
    
    def _init_layers(self):
        self.input_layer  = np.array(self.sim.AddNeuronLayer(self.n_inputs, 'InputSpikeNeuronModel', output_activity=True))
        self.action_layer = np.array(self.sim.AddNeuronLayer(self.n_total_channels, 'LIFTimeDrivenModel', self.action_neuron_params, output_activity=True))
        self.reward_layer = np.array(self.sim.AddNeuronLayer(1, 'InputSpikeNeuronModel', output_activity=True))
        self.punish_layer = np.array(self.sim.AddNeuronLayer(1, 'InputSpikeNeuronModel', output_activity=True))
    
    def _init_edlut_rules(self):
        self.sim.SetSimulationParameters({'num_simulation_queues': 4})

        # Create MSN D1 and D2 rules for the channels
        self.stde_d1_learning_rule_params = {
            **self.stde_d1_learning_rule_params,
            'prepos_tri_dif': [v*self.d1_learning_rate for v in self.stde_d1_learning_rule_params['prepos_tri_dif']],
            'pospre_tri_dif': [v*self.d1_learning_rate for v in self.stde_d1_learning_rule_params['pospre_tri_dif']],
            'pre_dif': self.stde_d1_learning_rule_params['pre_dif']*self.d1_learning_rate,
            'pos_dif': self.stde_d1_learning_rule_params['pos_dif']*self.d1_learning_rate,
            'prepos_dif': self.stde_d1_learning_rule_params['prepos_dif']*self.d1_learning_rate,
            'pospre_dif': self.stde_d1_learning_rule_params['pospre_dif']*self.d1_learning_rate
        }
        self.stde_d2_learning_rule_params = {
            **self.stde_d2_learning_rule_params,
            'prepos_tri_dif': [v*self.d2_learning_rate for v in self.stde_d2_learning_rule_params['prepos_tri_dif']],
            'pospre_tri_dif': [v*self.d2_learning_rate for v in self.stde_d2_learning_rule_params['pospre_tri_dif']],
            'pre_dif': self.stde_d2_learning_rule_params['pre_dif']*self.d2_learning_rate,
            'pos_dif': self.stde_d2_learning_rule_params['pos_dif']*self.d2_learning_rate,
            'prepos_dif': self.stde_d2_learning_rule_params['prepos_dif']*self.d2_learning_rate,
            'pospre_dif': self.stde_d2_learning_rule_params['pospre_dif']*self.d2_learning_rate
        }
        if not self.with_ach:
            self.stde_d1_learning_rule_params['ach_tri_dif'] = [1.0]*len(self.stde_d1_learning_rule_params['ach_tri_dif'])
            self.stde_d2_learning_rule_params['ach_tri_dif'] = [1.0]*len(self.stde_d2_learning_rule_params['ach_tri_dif'])


        self.edlut_stde_d1_learning_rule = self.sim.AddLearningRule('ESTDE', self.stde_d1_learning_rule_params)
        self.stde_d1_learning_rule = {**self.default_synapse_params, 'wchange': self.edlut_stde_d1_learning_rule}
        self.trigger_stde_d1_learning_rule = {**self.default_synapse_params, 'trigger_wchange': self.edlut_stde_d1_learning_rule}

        self.edlut_stde_d2_learning_rule = self.sim.AddLearningRule('ESTDE', self.stde_d2_learning_rule_params)
        self.stde_d2_learning_rule = {**self.default_synapse_params, 'wchange': self.edlut_stde_d2_learning_rule}
        self.trigger_stde_d2_learning_rule = {**self.default_synapse_params, 'trigger_wchange': self.edlut_stde_d2_learning_rule}

        self.edlut_vogels_learning_rule = self.sim.AddLearningRule('ESTDE', self.vogels_learning_rule_params)
        self.vogels_learning_rule = {**self.inhibitory_synapse_params, 'wchange': self.edlut_vogels_learning_rule, 'max_weight': 1e6}
