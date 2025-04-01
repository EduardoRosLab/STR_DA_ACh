#!/usr/bin/python

from pyedlut import simulation_wrapper as pyedlut
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Declare the simulation object
simulation = pyedlut.PySimulation_API()


n_inputs = 500
n_outputs = 1

total_simulation_time = 1000.0
simulation_step = 0.01

pattern_steps = 50
input_frequency = 10.0


def random_pattern():
    global n_inputs, simulation_step, pattern_steps, input_frequency
    pattern_duration = pattern_steps*simulation_step

#     n_spikes_per_neuron = int(input_frequency * pattern_duration)
#     oi = np.repeat(np.arange(n_inputs), n_spikes_per_neuron)
#     ot = np.random.rand(n_inputs*n_spikes_per_neuron) * pattern_duration

    n_spikes = int(input_frequency * pattern_duration * n_inputs)
    oi = np.random.randint(n_inputs, size=n_spikes)
    ot = np.random.rand(n_spikes) * pattern_duration    
    
    return ot, oi

repeating_pattern = random_pattern()

# Create the input neuron layers (three input fibers for spikes and, input fiber for current and a sinusoiidal current generator)
input_spike_layer = simulation.AddNeuronLayer(
        num_neurons=n_inputs,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)

# Get the default parameter values of the neuron model
default_params = simulation.GetNeuronModelDefParams('AsabukiNeuron');

# Get the default parameter values of the integration method
default_im_param = simulation.GetIntegrationMethodDefParams('Euler')

# Define the neuron model parameters for the output layer
integration_method = pyedlut.PyModelDescription(model_name='Euler', params_dict={'step': 0.001})
output_params = {
    'g_d': 0.7,
    'tau': 15.0,
    'tau_exc': 50.0, #5.0, #50.0,
    'tau_inh': 5.0,
    'tau_nmda': 50.0,
    'tau_ref': 2.0,
    
    'beta_0': 5.0,
    'phi_0': 10.0, #10.0,
    't_0': 3000.0,
    'theta_0': 1.7,
    'int_meth': integration_method
}


# Create the output layer
output_layer = simulation.AddNeuronLayer(
        num_neurons = n_outputs,
        model_name = 'AsabukiNeuron',
        param_dict = output_params,
        log_activity = False,
        output_activity = True
)


# Get the default parameter values of the neuron model
# default_param_lrule = simulation.GetLearningRuleDefParams('AsabukiSynapse');

# Define the learning rule parameters
lrule_params = {
    'eta': 1e-6, #5e-8, #5e-6,  
    'gamma': 5.0, #5.0
}

# Create the learning rule
eprop_rule = simulation.AddLearningRule('AsabukiSynapse', lrule_params)


# Define the synaptic parameters

src = []
tgt = []
for i in range(n_inputs):
    for j in range(n_outputs):
        src.append(input_spike_layer[i])
        tgt.append(output_layer[j])

connection_type = [0] * len(src)
wchange = [eprop_rule] * len(src)
synaptic_params = {
    'weight': (50.0 / (n_inputs*output_params['tau_exc'])) * np.random.rand(len(src)),
    'max_weight': 100.0,
    'type': connection_type,
    'delay': 0.001,
    'wchange': wchange,
    'trigger_wchange': -1
}

# Create the synapses
synaptic_layer = simulation.AddSynapticLayer(src, tgt, synaptic_params)

# Initialize the network
simulation.Initialize()


# Run the simulation step-by-step
sim_time = 0.0
ws = []
times = []

is_repeating_pattern = False
pattern = None


for step, sim_time in enumerate(tqdm(np.arange(0, total_simulation_time, simulation_step))):
    
    # Set the input pattern
    if (step % pattern_steps) == 0:
        if is_repeating_pattern:
            pattern = repeating_pattern
        else:
            pattern = random_pattern()
        is_repeating_pattern = not is_repeating_pattern
        
        spk_times = (pattern[0]+sim_time).tolist()
        spk_indices = (pattern[1]+input_spike_layer[0]).tolist()
        simulation.AddExternalSpikeActivity(spk_times, spk_indices)

    
    ws.append(simulation.GetWeights())
    times.append(sim_time)
    simulation.RunSimulation(sim_time + simulation_step)

ws.append(simulation.GetWeights())
times.append(sim_time)
    
print('Simulation finished')

# Retrieve output spike activity
output_times, output_index = simulation.GetSpikeActivity()

ot = np.array(output_times)
oi = np.array(output_index)
ws = np.array(ws)

_ = plt.plot(times, ws[:,::10])
plt.show()
