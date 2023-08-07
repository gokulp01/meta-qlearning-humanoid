import numpy as np
import torch
from collections import deque

class Runner:
    def __init__(self, env, model, replay_buffer=None, tasks_buffer=None, burn_in=1e4,
                 expl_noise=0.1, total_timesteps=1e6, max_path_length=200, history_length=1,
                 device='cpu'):
        self.model = model
        self.env = env
        self.burn_in = burn_in
        self.device = device
        self.episode_rewards = deque(maxlen=10)
        self.episode_lens = deque(maxlen=10)
        self.replay_buffer = replay_buffer
        self.expl_noise = expl_noise
        self.total_timesteps = total_timesteps
        self.max_path_length = max_path_length
        self.hist_len = history_length
        self.tasks_buffer = tasks_buffer 

    def run(self, update_iter, keep_burning=False, task_id=None, early_leave=200):
        obs = self.env.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0
        uiter = 0
        reward_epinfos = []
        rewards_hist = deque(maxlen=self.hist_len)
        actions_hist = deque(maxlen=self.hist_len)
        obsvs_hist = deque(maxlen=self.hist_len)

        next_hrews = deque(maxlen=self.hist_len)
        next_hacts = deque(maxlen=self.hist_len)
        next_hobvs = deque(maxlen=self.hist_len)

        zero_action = np.zeros(self.env.action_space.shape[0])
        zero_obs = np.zeros(obs.shape)
        for _ in range(self.hist_len):
            rewards_hist.append(0)
            actions_hist.append(zero_action.copy())
            obsvs_hist.append(zero_obs.copy())

            next_hrews.append(0)
            next_hacts.append(zero_action.copy())
            next_hobvs.append(zero_obs.copy())

        rand_action = np.random.normal(0, self.expl_noise, size=self.env.action_space.shape[0])
        rand_action = np.clip(rand_action, self.env.action_space.low, self.env.action_space.high)
        rewards_hist.append(0)
        actions_hist.append(rand_action.copy())
        obsvs_hist.append(obs.copy())

        while not done and uiter < min(self.max_path_length, early_leave):
            np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten()  
            np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)
            np_pre_obsers = np.asarray(obsvs_hist, dtype=np.float32).flatten()  

            if keep_burning or update_iter < self.burn_in:
                action = self.env.action_space.sample()

            else:
                action = self.model.select_action(np.array(obs), np.array(np_pre_actions),
                                                  np.array(np_pre_rewards), np.array(np_pre_obsers))

                if self.expl_noise != 0: 
                    action += np.random.normal(0, self.expl_noise, size=self.env.action_space.shape[0])
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            new_obs, reward, done, _ = self.env.step(action) 
            done_bool = float(done) if episode_timesteps + 1 != self.max_path_length else 0

            episode_reward += reward
            reward_epinfos.append(reward)

            next_hrews.append(reward)
            next_hacts.append(action.copy())
            next_hobvs.append(obs.copy())

            np_next_hacts = np.asarray(next_hacts, dtype=np.float32).flatten()
           
            np_next_hrews = np.asarray(next_hrews, dtype=np.float32)
            np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()

            replay_data = (obs, new_obs, action, reward, done_bool, np_pre_actions, np_pre_rewards,
                        np_pre_obsers, np_next_hacts, np_next_hrews, np_next_hobvs)

            self.replay_buffer.add(replay_data)
            self.tasks_buffer.add(task_id, replay_data)

            rewards_hist.append(reward)
            actions_hist.append(action.copy())
            obsvs_hist.append(obs.copy())

            obs = new_obs.copy()
            episode_timesteps += 1
            update_iter += 1
            uiter += 1

        info = {
            'episode_timesteps': episode_timesteps,
            'update_iter': update_iter,
            'episode_reward': episode_reward,
            'epinfos': [{'r': round(sum(reward_epinfos), 6), 'l': len(reward_epinfos)}]
        }

        return info
