from tqdm.auto import tqdm


class Task_Base:
    
    def __init__(self, **kwargs):
        self.progress_bar = kwargs.get('progress_bar', True)
        self.current_time = 0.0
        self.time_step = 10e-3 # each simulation step will be 10 ms

    def _init_vars(self):
        pass

    def run(self, as_generator=False):
        if as_generator:
            return self._run_generator()
        else:
            self._run_normal()
    
    def _run_normal(self):
        for i_trial, trial in enumerate(tqdm(self.experiment_design, ncols=500, disable=not self.progress_bar)):
            self._run_step(i_trial, trial)

    def _run_generator(self):                
        for i_trial, trial in enumerate(tqdm(self.experiment_design, ncols=500, disable=not self.progress_bar)):
            self._run_step(i_trial, trial)
            yield self.current_time, i_trial, trial

    def _run_step(self, i_trial, trial):
        # Inter-trial changes: reset rewards and retrieve weights
        rewarded_trial = False

        for i_cha, channel in enumerate(self.channels):
            w_d1, w_d2, w_fsi = channel.get_weights()
            self.channel_d1_weights[i_cha].append(w_d1)
            self.channel_d2_weights[i_cha].append(w_d2)
            self.channel_fsi_weights[i_cha].append(w_fsi)
        self.trial_times.append(self.current_time)

        for trial_step in range(self.n_trial_steps):
            
            # Intra-trial changes: update inputs
            self._update_inputs(i_trial, trial, trial_step)

            # Intra-trial steps
            for sim_step in range(self.trial_step_duration):                    
                
                # Next simulation step
                self.current_time += self.time_step
                self.sim.RunSimulation(self.current_time)

                # Get the spike activity
                spks = self.sim.GetSpikeActivity()
                self.ot += spks[0]
                self.oi += spks[1]
                
                # Process the rewards
                if not rewarded_trial:
                    reward_given = self._process_rewards(i_trial, trial, trial_step, spks) # If an action is done, generate a reward if needed
                    rewarded_trial = reward_given!=0
    
    def _update_inputs(self, i_trial, trial, trial_step):
        pass
    def _process_rewards(self, i_trial, trial, trial_step, spks):
        pass