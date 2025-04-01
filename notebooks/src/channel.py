from edlut_python_2_rules import simulation_wrapper as pyedlut
import numpy as np

class Channel:

    def __init__(self, **kwparams):        
        # Attributes that need to be initialized
        kwparams.get('network')
        self.attribution_neuron = kwparams.get('action_neuron')

        self._init_consts()

        for key, value in kwparams.items():
            setattr(self, key, value)
        self.kwparams = kwparams.copy() if type(kwparams) is dict else None
    
        self._init_models()
        self._init_network()


    def _init_consts(self):
        self.n_outputs = 10
        self.n_input_connections = -1

        self.lateral_weight_d1d1 = 0.0  #1.0,
        self.lateral_weight_d1d2 = 5.0  #1.0,
        self.lateral_weight_d2d1 = 10.0  #1.0,
        self.lateral_weight_d2d2 = 0.0  #1.0,

        self.d1_da_weight = 0.0 
        self.d2_da_weight = 100.0

        self.max_delay = 1.0/15.0
        self.delay_values = np.linspace(1e-3, self.max_delay, 10)

        self.fsi_init_weight = 10.0


    def _init_models(self, params={}):

        if getattr(self.network, 'sim', None) is None:
            print("Es recomendable usar el mismo simulador para todos los modelos.")
            self.network.sim = pyedlut.PySimulation_API()
            self.network.sim.SetSimulationParameters(params)    
        
        if getattr(self, 'integration_method', None) is None: #TODO: Add an integration method to the params
            self.integration_method = pyedlut.PyModelDescription(
                model_name='RK4',
                params_dict={'step': 5e-4})

        self.msn_d1_neuron_params = {**self.network.msn_d1_neuron_params, 'int_meth': self.integration_method}
        self.msn_d2_neuron_params = {**self.network.msn_d2_neuron_params, 'int_meth': self.integration_method}

        if getattr(self.network, 'stde_d1_learning_rule', None) is None:
            self.network.edlut_stde_d1_learning_rule = self.network.sim.AddLearningRule('ESTDE', self.network.stde_d1_learning_rule_params)
            self.network.stde_d1_learning_rule = {**self.default_synapse_params, 'wchange': self.network.edlut_stde_d1_learning_rule}
            self.network.trigger_stde_d1_learning_rule = {**self.network.default_synapse_params, 'trigger_wchange': self.network.stde_d1_learning_rule}
        if getattr(self.network, 'stde_d2_learning_rule', None) is None:
            self.network.edlut_stde_d2_learning_rule = self.network.sim.AddLearningRule('ESTDE', self.network.stde_d2_learning_rule_params)
            self.network.stde_d2_learning_rule = {**self.default_synapse_params, 'wchange': self.network.edlut_stde_d2_learning_rule}
            self.network.trigger_stde_d2_learning_rule = {**self.network.default_synapse_params, 'trigger_wchange': self.network.stde_d2_learning_rule}
        if getattr(self.network, 'vogels_learning_rule', None) is None:
            self.network.edlut_vogels_learning_rule = self.network.sim.AddLearningRule('ESTDE', self.network.vogels_learning_rule_params)
            self.network.vogels_learning_rule = {**self.network.inhibitory_synapse_params, 'wchange': self.network.edlut_vogels_learning_rule}


    def _init_network(s):

        # Create layers

        s.msn_d1_layer = s.network.sim.AddNeuronLayer(
            num_neurons = s.n_outputs,
            model_name = 'AdExTimeDrivenModel',
            param_dict = s.network.msn_d1_neuron_params,
            log_activity = False,
            output_activity = True
        )

        s.msn_d2_layer = s.network.sim.AddNeuronLayer(
            num_neurons = s.n_outputs,
            model_name = 'AdExTimeDrivenModel',
            param_dict = s.network.msn_d2_neuron_params,
            log_activity = False,
            output_activity = True
        )

        s.fsi_layer = s.network.sim.AddNeuronLayer(
            num_neurons = 1,
            model_name = 'PoissonGeneratorDeviceVector',
            param_dict = {'frequency': 500.0},
            log_activity = False,
            output_activity = False
        )
        
        # Input -> STR D1

        s.weights_d1 = []
        sources, targets = [], []
        for j in s.msn_d1_layer:
            if s.n_input_connections<0 or s.n_input_connections==len(s.network.input_layer):
                input_subset = s.network.input_layer
                s.n_input_connections = len(s.network.input_layer)
            else:
                input_subset = np.random.choice(s.network.input_layer, size=min(len(s.network.input_layer),s.n_input_connections), replace=False)
            for i in input_subset:
                sources.append(i)
                targets.append(j)

        weights = ((np.random.rand(len(sources)) * (s.network.max_init_weight_d1-s.network.min_init_weight_d1)) + s.network.min_init_weight_d1).tolist()
        delays = np.random.choice(s.delay_values, size=len(sources)).tolist()
        params = {
            **s.network.stde_d1_learning_rule,
            'weight': weights,
            'delay': delays,
            'max_weight': s.network.max_weight,
        }
        s.weights_d1 = s.network.sim.AddSynapticLayer(sources, targets, params)

        # Input -> STR D2

        s.weights_d2 = []
        sources, targets = [], []
        for j in s.msn_d2_layer:
            if s.n_input_connections<0 or s.n_input_connections==len(s.network.input_layer):
                input_subset = s.network.input_layer
            else:
                input_subset = np.random.choice(s.network.input_layer, size=min(len(s.network.input_layer),s.n_input_connections), replace=False)
            for i in input_subset:
                sources.append(i)
                targets.append(j)

        weights = ((np.random.rand(len(sources)) * (s.network.max_init_weight_d2-s.network.min_init_weight_d2)) + s.network.min_init_weight_d2).tolist()
        delays = np.random.choice(s.delay_values, size=len(sources)).tolist()
        params = {
            **s.network.stde_d2_learning_rule,
            'weight': weights,
            'delay': delays,
            'max_weight': s.network.max_weight,
        }
        s.weights_d2 = s.network.sim.AddSynapticLayer(sources, targets, params)

        # Lateral inhibition between and within layers

        connections = [
            (s.msn_d1_layer, s.msn_d1_layer, s.lateral_weight_d1d1),
            (s.msn_d1_layer, s.msn_d2_layer, s.lateral_weight_d1d2),
            (s.msn_d2_layer, s.msn_d1_layer, s.lateral_weight_d2d1),
            (s.msn_d2_layer, s.msn_d2_layer, s.lateral_weight_d2d2),
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
                _ = s.network.sim.AddSynapticLayer(
                    sources, targets,
                    {**s.network.inhibitory_synapse_params, 'weight': weights, 'max_weight': np.max(weights)})


        # DA D1 trigger conns
        _ = s.network.sim.AddSynapticLayer(
            [s.network.reward_layer[0]]*len(s.msn_d1_layer), s.msn_d1_layer, 
            {**s.network.trigger_stde_d1_learning_rule, 'type':0, 'delay': 1e-2})
        _ = s.network.sim.AddSynapticLayer(
            [s.network.punish_layer[0]]*len(s.msn_d1_layer), s.msn_d1_layer, 
            {**s.network.trigger_stde_d1_learning_rule, 'type':1, 'delay': 1e-2})
        # DA D2 trigger conns
        _ = s.network.sim.AddSynapticLayer(
            [s.network.reward_layer[0]]*len(s.msn_d2_layer), s.msn_d2_layer, 
            {**s.network.trigger_stde_d2_learning_rule, 'type':0, 'delay': 1e-2})
        _ = s.network.sim.AddSynapticLayer(
            [s.network.punish_layer[0]]*len(s.msn_d2_layer), s.msn_d2_layer, 
            {**s.network.trigger_stde_d2_learning_rule, 'type':1, 'delay': 1e-2})

        # DA D1 excitatory/inhibitory conns
        _ = s.network.sim.AddSynapticLayer(
            [s.network.reward_layer[0]]*len(s.msn_d1_layer), s.msn_d1_layer, 
            {**s.network.excitatory_synapse_params, 'max_weight': 1e6, 'weight': s.d1_da_weight})        
        # HACK: La parte inhibitoria no está habilitada.
        # _ = s.network.sim.AddSynapticLayer([s.punish_layer[0]]*len(s.msn_d1_layer), s.msn_d1_layer, {**s.inhibitory_params, 'max_weight': 1e6, 'weight': s.d1_da_weight})        

        # DA D2 inhibitory/excitatory conns
        # HACK: La parte excitatoria no está habilitada.
        # _ = s.network.sim.AddSynapticLayer([s.reward_layer[0]]*len(s.msn_d2_layer), s.msn_d2_layer, {**s.inhibitory_params, 'max_weight': 1e6, 'weight': s.d2_da_weight})        
        _ = s.network.sim.AddSynapticLayer(
            [s.network.punish_layer[0]]*len(s.msn_d2_layer), s.msn_d2_layer, 
            {**s.network.excitatory_synapse_params, 'max_weight': 1e6, 'weight': s.d2_da_weight})        

        # ACh trigger conns

        _ = s.network.sim.AddSynapticLayer(
            [s.attribution_neuron]*len(s.msn_d1_layer), s.msn_d1_layer, 
            {**s.network.trigger_stde_d1_learning_rule, 'type':2})
        _ = s.network.sim.AddSynapticLayer(
            [s.attribution_neuron]*len(s.msn_d2_layer), s.msn_d2_layer, 
            {**s.network.trigger_stde_d2_learning_rule, 'type':2})
        
        # FSI inhibitory conns
        s.weights_fsi = s.network.sim.AddSynapticLayer(
            s.fsi_layer * len(s.msn_d1_layer + s.msn_d2_layer), 
            s.msn_d1_layer + s.msn_d2_layer, 
            {
                **s.network.vogels_learning_rule,
                'weight': s.fsi_init_weight,
            })
    

    def get_weights(self):
        w_d1 = self.network.sim.GetSelectedWeights(self.weights_d1)
        w_d2 = self.network.sim.GetSelectedWeights(self.weights_d2)
        w_fsi = self.network.sim.GetSelectedWeights(self.weights_fsi)
        return w_d1, w_d2, w_fsi
