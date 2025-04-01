import numpy as np


class Params_Base:
    def __init__(self, **kwargs) -> None:
        self.default_neuron_params = {
            'c_m': 250.0,
            'e_exc': 0.0,
            'e_inh': -85.0,
            'e_leak': -65.0,
            'g_leak': 25.0,
            'tau_exc': 5.0,
            'tau_inh': 10.0,
            'tau_nmda': 20.0,
            'tau_ref': 1.0,
            'v_thr': -40.0,
        }

        self.check_neuron_params = {
            **self.default_neuron_params,
            'c_m': 10.0,
            'g_leak': 0.2,
            'tau_ref': 1000.0/1.0,
            'tau_exc': 50.0, 
            'tau_inh': 50.0, 
        }

        self.msn_d1_neuron_params = {
            'a': -14.5,
            'b': 500.0,
            'c_m': 123.5,
            'e_exc': 0.0,
            'e_inh': -80.0,
            'e_leak': -96.2,
            'e_reset': -51.0,
            'g_leak': 35.0,
            'tau_exc': 5.0,
            'tau_inh': 3.0,
            'tau_nmda': 20.0,
            'tau_w': 15.0,
            'thr_slo_fac': 16.0,
            'v_thr': -51.0,
        }

        self.msn_d2_neuron_params = {
            'a': -15.0,
            'b': 350.0,
            'c_m': 102.3,
            'e_exc': 0.0,
            'e_inh': -80.0,
            'e_leak': -84.6,
            'e_reset': -49.0,
            'g_leak': 36.54,
            'tau_exc': 5.0,
            'tau_inh': 3.0,
            'tau_nmda': 20.0,
            'tau_w': 17.0,
            'thr_slo_fac': 16.0,
            'v_thr': -49.0,
        }

        self.action_neuron_params = {
            'c_m': 100.0, #15.0,
            'e_exc': 0.0,
            'e_inh': -80.0,
            'e_leak': -65.0,
            'g_leak': 5e-2, #0.2,
            'tau_exc': 5.0,
            'tau_inh': 10.0,
            'tau_nmda': 20.0,
            'tau_ref': 100.0,
            'v_thr': -50.0
        }

        self.default_synapse_params = {
            'weight': 1e-2,
            'max_weight': 10.0,
            'type': 0,
            'delay': 0.001,
            'wchange': -1,
            'trigger_wchange': -1,
        }

        self.excitatory_synapse_params = {**self.default_synapse_params,  'type': 0}
        self.inhibitory_synapse_params = {**self.default_synapse_params,  'type': 1}
        self.current_synapse_params = {**self.default_synapse_params, 'type': 3}


        self.stde_d1_learning_rule_params = {
            'tau_pre': 0.032,
            'tau_pos': 0.032,
            'tau_eli': 0.5,
            'tau_ach': 0.2,
            'ach_tri_dif': [0.0, 0.0, 3.0, 1.0],
            'prepos_tri_dif': [10.0, -5.0, 0.0, 0.0],
            'pospre_tri_dif': [0.0, -5.0, 0.0, 0.0],
            #'prepos_tri_dif': [ 10.0, -10.0, 0.0, 0.0],
            #'pospre_tri_dif': [-10.0,  10.0, 0.0, 0.0],
            'pre_dif': 0.0,
            'pos_dif': 0.0,
            'prepos_dif': 1e-2,
            'pospre_dif': -1.5e-2
        }

        self.stde_d2_learning_rule_params = {
            'tau_pre': 0.032,
            'tau_pos': 0.032,
            'tau_eli': 0.5,
            'tau_ach': 0.2,
            'ach_tri_dif': [0.0, 0.0, 3.0, 1.0],
            # 'prepos_tri_dif': [-10.0, 5.0, 0.0, 0.0],
            # 'pospre_tri_dif': [0.0, 5.0, 0.0, 0.0],
            'prepos_tri_dif': [-2.0, 5.0, 0.0, 0.0],
            'pospre_tri_dif': [0.0, 5.0, 0.0, 0.0],
            'pre_dif': 0.0,
            'pos_dif': -0.5e-2,
            'prepos_dif': 1e-2,
            'pospre_dif': -1.5e-2 #0.0
        }

        self.d1_learning_rate = 1e-3
        self.d2_learning_rate = 1e-3
        self.with_ach = True


        # Sinapsis FSI -> MSN

        target_firing_rate = 10.0
        lr = 1e-3

        prepos_dif = lr #lr
        tau_pre = tau_pos = 0.1 #0.02 #0.032
        pospre_dif = lr
        tau_eli = 0.6
        tau_ach = 0.1
        prepos_tri_dif = [0.0, 0.0, 0.0]
        pospre_tri_dif = [0.0, 0.0, 0.0]
        ach_tri_dif = [0.0, 0.0, 0.0]

        pre_dif = -lr * 2.0 * target_firing_rate * tau_pos;
        pos_dif = 0.0

        self.vogels_learning_rule_params = {
            'pre_dif': pre_dif,
            'prepos_dif': prepos_dif,
            'tau_pre': tau_pre,
            'pos_dif': pos_dif,
            'pospre_dif': pospre_dif,
            'tau_pos': tau_pos,
            'tau_eli': tau_eli,
            'tau_ach': tau_ach,
            'prepos_tri_dif': prepos_tri_dif,
            'pospre_tri_dif': pospre_tri_dif,
            'ach_tri_dif': ach_tri_dif,
        }



def generate_poisson_spike_times(interval_duration, firing_rate):
    '''
    Generates spike times following a Poisson distribution, given an interval duration and a firing rat.

    # Example usage:
    interval_duration = 0.1  # 0.5 seconds
    firing_rate = 5  # 5 Hz (spikes per second)

    spike_times = generate_poisson_spike_times(interval_duration, firing_rate)
    print("Spike times:", spike_times)
    '''

    # Calculate the expected number of spikes in the interval
    expected_num_spikes = interval_duration * firing_rate

    # Sample the number of spikes from a Poisson distribution
    num_spikes = np.random.poisson(expected_num_spikes)

    # Generate uniformly distributed spike times within the interval duration
    spike_times = np.random.uniform(0, interval_duration, num_spikes)

    # Sort the spike times in ascending order
    # spike_times.sort()

    return spike_times#.tolist()


def generate_poisson_spike_times_vector(interval_duration, firing_rates):
    expected_num_spikes = interval_duration * firing_rates
    num_spikes = np.random.poisson(expected_num_spikes, size=firing_rates.size)
    spike_times = np.random.uniform(0, interval_duration, size=np.sum(num_spikes))
    sub = num_spikes>0
    arg_sub = np.where(sub)[0]
    spike_emitters = np.repeat(arg_sub, num_spikes[sub])
    return spike_times, spike_emitters