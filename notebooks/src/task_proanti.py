import numpy as np
from itertools import product as cartesian_product
from task_base import Task_Base
from misc import generate_poisson_spike_times_vector


class Encoding_Task_ProAnti:
    def __init__(self, **kwargs):
        self.n_retina_neurons = 101
        self.retina_size = 5

        self.trial_start = -0.5
        self.trial_stop = 0.5

        self.trial_types = ['P', 'A']
        self.stimulus_positions = ['L', 'R']
        self.target_positions = ['L', 'R']
        self.all_trial_cases = list(cartesian_product(
            self.trial_types,
            self.stimulus_positions,
            self.target_positions
        ))

        self.n_task_inputs = self.n_retina_neurons # Number of inputs that contains task information

        self.time_step = 10e-3 if getattr(self, 'time_step', None) is None else self.time_step

        # Update the attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.total_trial_time = self.trial_stop - self.trial_start
        self.trial_times = np.arange(self.trial_start, self.trial_stop, step=self.time_step)

        self.create_signal_trapezoids()
        self.pre_cache_trials()
    def create_signal_trapezoids(self):
        self.signals = {}
        self.signals['A']  = trapezoid_func( 0e-3,   50e-3,  200e-3,  250e-3, 0.0, 1.0)
        self.signals['B']  = trapezoid_func( 60.0e-3,   160.0e-3,  188.0e-3,  200.0e-3, 0.0, 1.0)
        self.signals['C']  = trapezoid_func(-500.0e-3, -440.0e-3, -170.0e-3, -70.0e-3,  0.0, 1.0)
        self.signals['D']  = trapezoid_func( 0e-3,  50e-3,  200e-3,  250e-3, 0.0, 1.0)
        self.signals['E1'] = trapezoid_func(-500.0e-3, -440.0e-3,  140.0e-3,  200.0e-3, 0.0, 1.0)
        self.signals['E2'] = trapezoid_func( 200.0e-3,  260.0e-3,  500.0e-3,  560.0e-3, 0.0, 1.0)
        self.signals['F']  = trapezoid_func(-500.0e-3, -100.0e-3,  188.0e-3,  200.0e-3, 0.0, 1.0)
        self.signals['G']  = trapezoid_func( 120.0e-3,  200.0e-3,  200.0e-3,  216.0e-3, 0.0, 1.0)
        self.signals['H1'] = trapezoid_func(-500.0e-3, -490.0e-3,  490.0e-3,  500.0e-3, 0.0, 1.0)
        self.signals['H2'] = trapezoid_func( 120.0e-3,  200.0e-3,  200.0e-3,  280.0e-3, 0.0, 1.0)
        self.signals['Task'] = trapezoid_func( -500e-3, -400e-3,  400e-3, 500e-3, 0.0, 1.0)
    def pre_cache_trials(self):
        self._trial = {}
        self.trial = {}

        for task, sti_pos, tgt_pos in self.all_trial_cases:
            self._trial[task] = self._trial.get(task, {})
            self._trial[task][sti_pos] = self._trial[task].get(sti_pos, {})
            self._trial[task][sti_pos][tgt_pos] = self._trial[task][sti_pos].get(tgt_pos, {})

            self.trial[task] = self.trial.get(task, {})
            self.trial[task][sti_pos] = self.trial[task].get(sti_pos, {})

            _task = 2 if task == 'P' else -2
            _sti_pos = 2 if sti_pos == 'R' else -2
            _tgt_pos = 2 if tgt_pos == 'R' else -2

            self._trial[task][sti_pos][tgt_pos]['A'] = np.outer(    #sti
                k_func(_sti_pos, self.n_retina_neurons, self.n_retina_neurons, retina_size=self.retina_size), 
                self.signals['A'](self.trial_times))
            self._trial[task][sti_pos][tgt_pos]['B'] = np.outer(    #sti
                k_func(_sti_pos, self.n_retina_neurons, retina_size=self.retina_size), 
                self.signals['B'](self.trial_times))
            self._trial[task][sti_pos][tgt_pos]['C'] = np.outer(
                k_func(0, self.n_retina_neurons, retina_size=self.retina_size), 
                self.signals['C'](self.trial_times))
            self._trial[task][sti_pos][tgt_pos]['D'] = np.outer(    #tgt
                k_func(_tgt_pos, self.n_retina_neurons, retina_size=self.retina_size), 
                self.signals['D'](self.trial_times))
            self._trial[task][sti_pos][tgt_pos]['E'] = np.outer(
                k_func(0, self.n_retina_neurons, retina_size=self.retina_size), 
                (  
                    self.signals['E1'](self.trial_times) + 
                    self.signals['E2'](self.trial_times)
                ))
            self._trial[task][sti_pos][tgt_pos]['F'] = np.outer(
                (
                    k_func(_sti_pos, self.n_retina_neurons, retina_size=self.retina_size) + 
                    k_func(-_sti_pos, self.n_retina_neurons, retina_size=self.retina_size)
                ), 
                self.signals['F'](self.trial_times))
            self._trial[task][sti_pos][tgt_pos]['G'] = np.outer(
                k_func(_tgt_pos, self.n_retina_neurons, retina_size=self.retina_size), 
                self.signals['G'](self.trial_times))
            self._trial[task][sti_pos][tgt_pos]['H'] = np.maximum(
                np.outer(
                    k_func(0, self.n_retina_neurons, retina_size=self.retina_size), 
                    self.signals['H1'](self.trial_times)),
                np.outer(
                    k_func(0, self.n_retina_neurons, retina_size=self.retina_size, gamma=100.0),
                    self.signals['H2'](self.trial_times)))
            self._trial[task][sti_pos][tgt_pos]['Task'] = np.outer( #task
                k_func(_task, self.n_retina_neurons, retina_size=self.retina_size), 
                self.signals['Task'](self.trial_times))

            self.trial[task][sti_pos][tgt_pos] = np.r_[
                self._trial[task][sti_pos][tgt_pos]['Task'],
                self._trial[task][sti_pos][tgt_pos]['A'],
                self._trial[task][sti_pos][tgt_pos]['B'],
                self._trial[task][sti_pos][tgt_pos]['C'],
                self._trial[task][sti_pos][tgt_pos]['D'],
                self._trial[task][sti_pos][tgt_pos]['E'],
                self._trial[task][sti_pos][tgt_pos]['F'],
                self._trial[task][sti_pos][tgt_pos]['G'],
                self._trial[task][sti_pos][tgt_pos]['H'],
            ]


class Task_ProAnti(Task_Base, Encoding_Task_ProAnti):
    
    def __init__(self, **kwargs):
        Task_Base.__init__(self, **kwargs)
        Encoding_Task_ProAnti.__init__(self, **kwargs)
        self.n_trials = 10
        self.n_trial_types = len(self.all_trial_cases)
        self.n_actions = 2

        self.n_trial_steps = len(self.trial_times)
        self.trial_step_duration = 1

        self.min_input_activity = 1.0
        self.max_input_activity = 10.0

        self.n_input_connections = -1 #250
    

    def _init_vars(self):
        choices = np.random.choice(len(self.all_trial_cases), size=self.n_trials)
        self.experiment_design = np.array([self.all_trial_cases[i] for i in choices])
        task, sti_pos, tgt_pos = self.experiment_design[0]
        self.n_inputs = self.trial[task][sti_pos][tgt_pos].shape[0]


    def _update_inputs(self, i_trial, trial, trial_step):
        task, sti_pos, tgt_pos = self.experiment_design[i_trial]
        self.current_input = self.trial[task][sti_pos][tgt_pos][:,trial_step]
        self.current_input = self.current_input * (self.max_input_activity - self.min_input_activity) + self.min_input_activity

        spk_times, spk_indices = generate_poisson_spike_times_vector(self.time_step, self.current_input)
        spk_times += self.current_time
        spk_indices += self.input_layer[0]

        self.sim.AddExternalSpikeActivity(spk_times.tolist(), spk_indices.tolist())


    def _process_rewards(self, i_trial, trial, trial_step, spks):
        reward = 0

        ot = np.array(spks[0])
        oi = np.array(spks[1])

        # Check if the action was done
        action_neurons_fired = np.unique(oi[np.in1d(oi, self.action_layer)])
        actions_done = np.where(np.in1d(self.action_layer[self.non_output_channels:], action_neurons_fired))[0]

        if (trial_step<self.n_trial_steps//2):
            if len(actions_done) > 0:
                reward -= 1

        if (trial_step>self.n_trial_steps//2):
            stimulus_position = 0 if trial[1]=='L' else 1
            for action_done in actions_done:
                action_position = action_done % 2
                rewarded_action = (trial[0]=='P' and stimulus_position==action_position) or (trial[0]=='A' and stimulus_position!=action_position)
                reward += 1 if rewarded_action else -1
        
        if reward > 0:
            self.sim.AddExternalSpikeActivity([self.current_time+300e-3], self.reward_layer)
        elif reward < 0:
            self.sim.AddExternalSpikeActivity([self.current_time+300e-3], self.punish_layer)
        
        return reward


@np.vectorize
def interpolate(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
@np.vectorize
def k_func(mu, N=100, gamma=0.6, retina_size=5.0):
    position = np.linspace(-retina_size, retina_size, num=N)
    dif = np.abs(position - mu)
    dis = np.exp(-(dif * dif) / (2 * gamma * gamma))
    return dis
@np.vectorize
def trapezoid(t, a, b, c, d, rest_val, peak_val):
    if t < a:  # pre-onset
        return rest_val
    elif t < b:  # start onset
        return interpolate(t, a, b, rest_val, peak_val)
    elif t < c:  # plateau
        return peak_val
    elif t < d:  # start outset
        return interpolate(t, c, d, peak_val, rest_val)
    else:
        return rest_val
def trapezoid_func(a, b, c, d, rest_val, peak_val):
    def tf(t):
        return trapezoid(t, a, b, c, d, rest_val, peak_val)
    return tf


if __name__ == '__main__':
    experiment = Encoding_Task_ProAnti()
