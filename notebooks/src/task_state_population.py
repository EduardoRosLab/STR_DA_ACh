import numpy as np

from misc import generate_poisson_spike_times
from task_base import Task_Base

class Task_StatePopulation(Task_Base):

    def __init__(self, **kwargs):        
        super().__init__(**kwargs)

        self.n_trials = 100 # Experiment length, in number of trials
        self.n_trial_types = 2
        self.n_actions = 2

        self.n_trial_steps = 5 # in number of different input populations used per trial.
        self.trial_step_duration = 40 # in number of time steps
        self.trial_width = 10 # in number of inputs used.
        self.trial_width_step = 5 # in number of neurons

        self.max_input_activity = 50.0
            

    def _init_vars(self):
        self.experiment_design = np.random.choice(range(self.n_trial_types), size=self.n_trials)
        self.n_inputs = self.n_trial_types * (self.n_trial_steps * self.trial_width_step + self.trial_width)


    def _update_inputs(self, i_trial, trial, trial_step):
        duration = self.trial_step_duration*self.time_step
        spk_times = generate_poisson_spike_times(duration, self.max_input_activity*self.trial_width)
        spk_times += self.current_time
        
        from_i = trial*(self.n_trial_steps*self.trial_width_step+self.trial_width) + trial_step*self.trial_width_step
        to_i = from_i + self.trial_width - 1
        spk_indices = np.random.choice(range(from_i,to_i+1), size=len(spk_times))
        
        self.sim.AddExternalSpikeActivity(spk_times.tolist(), spk_indices.tolist())


    def _process_rewards(self, i_trial, trial, trial_step, spks):
        reward = 0

        ot = np.array(spks[0])
        oi = np.array(spks[1])

        # Check if the action was done
        action_neurons_fired = np.unique(oi[np.in1d(oi, self.action_layer)])
        actions_done = np.where(np.in1d(self.action_layer, action_neurons_fired))[0]

        # Punishment if the agent responds too early
        if False:
            #if (trial_step>self.n_trial_steps//10 and trial_step<self.n_trial_steps//4):
            if trial_step < self.n_trial_steps//4:
                if len(actions_done) > 0:
                    reward = -1

        # Response only valid after half the trial
        if True:
            if (trial_step>self.n_trial_steps//2):
                for action_done in actions_done:
                    if ((action_done%self.n_trial_types) == (trial%self.n_actions)):
                        reward = 1 if reward >= 0 else -1
                    else:
                        reward = -1
        # Response valid at any moment
        else:
            for action_done in actions_done:
                if ((action_done%self.n_trial_types) == (trial%self.n_actions)):
                    reward = 1 if reward >= 0 else -1
                else:
                    reward = -1
        
        if reward > 0:
            self.sim.AddExternalSpikeActivity([self.current_time+3e-1], self.reward_layer)
        elif reward < 0:
            self.sim.AddExternalSpikeActivity([self.current_time+3e-1], self.punish_layer)
        
        return reward


class Task_StatePopulationRelearning(Task_StatePopulation):
    def _init_vars(self):        
        super()._init_vars()
        
        self.reward_policy = np.zeros((self.n_trial_types, self.n_actions))
        self.reward_policy[:] = -1
        for i in range(self.n_trial_types):
            self.reward_policy[i, i%self.n_actions] = 1


    def _process_rewards(self, i_trial, trial, trial_step, spks):
        reward = 0

        ot = np.array(spks[0])
        oi = np.array(spks[1])

        # Check if the action was done
        action_neurons_fired = np.unique(oi[np.in1d(oi, self.action_layer)])
        actions_done = np.where(np.in1d(self.action_layer, action_neurons_fired))[0]

        # Response only valid after a fraction of the trial
        if (trial_step>self.n_trial_steps//10):
            for action_done in actions_done:
                current_reward = self.reward_policy[trial, action_done]                
                if current_reward == -1:
                    reward = -1  # Castigo invalida todo
                    break        # Rompemos el bucle inmediatamente
                elif current_reward == 1:
                    reward = 1   # Se da recompensa si no hay castigo
            # `reward` se mantiene en 0 si no hay acciones realizadas o todas son neutras.
        
        if reward > 0:
            self.sim.AddExternalSpikeActivity([self.current_time+3e-1], self.reward_layer)
        elif reward < 0:
            self.sim.AddExternalSpikeActivity([self.current_time+3e-1], self.punish_layer)
        
        return reward