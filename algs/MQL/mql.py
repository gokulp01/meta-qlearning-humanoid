from __future__ import  print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from copy import deepcopy
from sklearn.linear_model import LogisticRegression as logistic

class MQL:
    def __init__(self, actor, actor_target, critic, critic_target, learning_rate=None, discount_factor=0.99,
                 polyak_tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_frequency=2, batch_size=100,
                 optimization_method='', max_action=None, max_iter_logistic=2000, beta_clip=1,
                 enable_beta_observation_context=False, proximity_coefficient=1, device='cpu', csc_lambda=1.0,
                 training_type='csc', use_ess_clipping=False, use_normalized_beta=True, reset_optimizers=False):
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.discount_factor = discount_factor
        self.polyak_tau = polyak_tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        self.batch_size = batch_size
        self.max_iter_logistic = max_iter_logistic
        self.beta_clip = beta_clip
        self.enable_beta_observation_context = enable_beta_observation_context
        self.proximity_coefficient = proximity_coefficient
        self.proximity_coefficient_initial = proximity_coefficient
        self.device = device
        self.csc_lambda = csc_lambda
        self.training_type = training_type
        self.use_ess_clipping = use_ess_clipping
        self.r_eps = np.float32(1e-7)
        self.use_normalized_beta = use_normalized_beta
        self.set_training_style()
        self.learning_rate = learning_rate
        self.reset_optimizers = reset_optimizers

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.copy_model_parameters()

        if learning_rate:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters())
            self.critic_optimizer = optim.Adam(self.critic.parameters())

        print(self.actor_optimizer)
        print(self.critic_optimizer)
        print('********')
        print(reset_optimizers)
        print(use_ess_clipping)
        print(use_normalized_beta)
        print(enable_beta_observation_context)
        print('********')

    def copy_model_parameters(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def set_training_style(self):
        if self.training_type == 'csc':
            self.train_actor = self.train_actor_csc
            self.train_critic = self.train_critic_csc
        elif self.training_type == 'ddpg':
            self.train_actor = self.train_actor_ddpg
            self.train_critic = self.train_critic_ddpg
        else:
            raise ValueError(f'Invalid training type: {self.training_type}')

    def train_actor_csc(self, states, beta, csc_rewards):
        self.actor_optimizer.zero_grad()

        actor_actions = self.actor(states)
        q_values = self.critic(torch.cat((states, actor_actions), dim=1))
        csc_loss = self.csc_lambda * (q_values * beta).mean()
        loss = -csc_rewards.mean() + csc_loss
        loss.backward()
       
        if self.proximity_coefficient > 0:
            prox_loss = self.proximity_coefficient * ((actor_actions - states) ** 2).mean()
            prox_loss.backward(retain_graph=True)

        self.actor_optimizer.step()
        self.proximity_coefficient *= 0.999

        return csc_loss.item(), -csc_rewards.mean().item()

    def train_critic_csc(self, states, actions, rewards, next_states, dones, beta, next_beta):
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(torch.cat((next_states, next_actions), dim=1))
            target_q_values = rewards + (1 - dones) * self.discount_factor * (next_q_values - next_beta * torch.log(beta + self.r_eps))

        q_values = self.critic(torch.cat((states, actions), dim=1))
        td_errors = (q_values - target_q_values.detach())
        csc_loss = self.csc_lambda * (q_values * beta).mean()
        loss = (td_errors ** 2).mean() + csc_loss
        loss.backward()

        self.critic_optimizer.step()

        return td_errors.abs().mean().item(), csc_loss.item()

    def train_actor_ddpg(self, states):
        self.actor_optimizer.zero_grad()

        actor_actions = self.actor(states)
        critic_values = self.critic(torch.cat((states, actor_actions), dim=1))
        loss = -critic_values.mean()
        loss.backward()

        self.actor_optimizer.step()

        return None, -critic_values.mean().item()

    def train_critic_ddpg(self, states, actions, rewards, next_states, dones):
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(torch.cat((next_states, next_actions), dim=1))
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        q_values = self.critic(torch.cat((states, actions), dim=1))
        td_errors = (q_values - target_q_values.detach())
        loss = (td_errors ** 2).mean()
        loss.backward()

        self.critic_optimizer.step()

        return td_errors.abs().mean().item(), None


# not worrking
    # def copy_model_params(self):

    #     self.ckpt = {
    #                     'actor': deepcopy(self.actor),
    #                     'critic': deepcopy(self.critic)
    #                 }

    # def set_tasks_list(self, tasks_idx):
    #     '''
    #         Keep copy of task lists
    #     '''
    #     self.train_tasks_list = set(tasks_idx.copy())

    # def select_action(self, obs, previous_action, previous_reward, previous_obs):

    #     '''
    #         return action
    #     '''
    #     obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
    #     previous_action = torch.FloatTensor(previous_action.reshape(1, -1)).to(self.device)
    #     previous_reward = torch.FloatTensor(previous_reward.reshape(1, -1)).to(self.device)
    #     previous_obs = torch.FloatTensor(previous_obs.reshape(1, -1)).to(self.device)

    #     # combine all other data here before send them to actor
    #     # torch.cat([previous_action, previous_reward], dim = -1)
    #     pre_act_rew = [previous_action, previous_reward, previous_obs]

    #     return self.actor(obs, pre_act_rew).cpu().data.numpy().flatten()

    def get_proximity_penalty(model_t, model_target):

        param_prox = []
        for p, q in zip(model_t.parameters(), model_target.parameters()):
            param_prox.append(torch.norm(p - q.detach()) ** 2)

        result = sum(param_prox)

        return result.item()
    def train_cs(self, task_id=None, snap_buffer=None, train_tasks_buffer=None, adaptation_step=False):

        if adaptation_step:
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list))) + 2
            neg_tasks_ids = self.train_tasks_list
        else:
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list) - 1)) + 2
            neg_tasks_ids = list(self.train_tasks_list.difference(set([task_id])))

        pu, pr, px, xx = train_tasks_buffer.sample(task_ids=neg_tasks_ids, batch_size=task_bsize)
        neg_actions = torch.FloatTensor(pu).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_rewards = torch.FloatTensor(pr).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_obs = torch.FloatTensor(px).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_xx = torch.FloatTensor(xx).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)

        ppu, ppr, ppx, pxx = snap_buffer.sample(task_ids=[task_id], batch_size=snap_buffer.size_rb(task_id))
        pos_actions = torch.FloatTensor(ppu).to(self.device)
        pos_rewards = torch.FloatTensor(ppr).to(self.device)
        pos_obs = torch.FloatTensor(ppx).to(self.device)
        pos_pxx = torch.FloatTensor(pxx).to(self.device)

        pos_act_rew_obs = [pos_actions, pos_rewards, pos_obs]
        neg_act_rew_obs = [neg_actions, neg_rewards, neg_obs]

        with torch.no_grad():
            if self.enable_beta_obs_cxt:
                snap_ctxt = torch.cat([pos_pxx, self.actor.get_conext_feats(pos_act_rew_obs)], dim=-1).cpu().data.numpy()
                neg_ctxt = torch.cat([neg_xx, self.actor.get_conext_feats(neg_act_rew_obs)], dim=-1).cpu().data.numpy()
            else:
                snap_ctxt = self.actor.get_conext_feats(pos_act_rew_obs).cpu().data.numpy()
                neg_ctxt = self.actor.get_conext_feats(neg_act_rew_obs).cpu().data.numpy()

        x = np.concatenate((snap_ctxt, neg_ctxt))
        y = np.concatenate((-np.ones(snap_ctxt.shape[0]), np.ones(neg_ctxt.shape[0])))

        model = logistic(solver='lbfgs', max_iter=self.max_iter_logistic, C=self.lam_csc).fit(x, y)
        prediction_score = model.score(x, y)

        info = (snap_ctxt.shape[0], neg_ctxt.shape[0], model.score(x, y))

        return model, info

    def update_prox_w_ess_factor(self, cs_model, x, beta=None):
        n = x.shape[0]
        if beta is not None:
            w = ((torch.sum(beta)**2) /(torch.sum(beta**2) + self.r_eps) )/n
            ess_factor = np.clip(w.numpy(), self.r_eps, np.inf)
        else:
            p0 = cs_model.predict_proba(x)[:,0]
            w =  p0 / ( 1 - p0 + self.r_eps)
            w = (np.sum(w)**2) / (np.sum(w**2) + self.r_eps)
            ess_factor = np.clip(w/n, self.r_eps, np.inf)

        ess_prox_factor = 1.0 - ess_factor

        if not (self.r_eps <= ess_prox_factor <= np.inf):
            self.prox_coef = self.prox_coef_init
        else:
            self.prox_coef = ess_prox_factor

    def get_propensity(self, cs_model, curr_pre_act_rew, curr_obs):
        with torch.no_grad():
            if self.enable_beta_obs_cxt:
                ctxt = torch.cat([curr_obs, self.actor.get_conext_feats(curr_pre_act_rew)], dim=-1).cpu().numpy()
            else:
                ctxt = self.actor.get_conext_feats(curr_pre_act_rew).cpu().numpy()

        f_prop = np.dot(ctxt, cs_model.coef_.T) + cs_model.intercept_
        f_prop = torch.from_numpy(f_prop).float().clamp(min=-self.beta_clip)
        f_score = torch.exp(-f_prop).clamp(min=0.0, max=1.0)

        if self.use_normalized_beta:
            lr_prob = cs_model.predict_proba(ctxt)[:, 0]
            d_pmax_pmin = np.max(lr_prob) - np.min(lr_prob)
            f_score = ((f_score - torch.min(f_score)) / (torch.max(f_score) - torch.min(f_score) + self.r_eps)) * d_pmax_pmin + np.min(lr_prob)

        if self.use_ess_clipping:
            self.update_prox_w_ess_factor(cs_model, ctxt, beta=f_score)

        return f_score, None


    def do_training(self,
                    replay_buffer = None,
                    iterations = None,
                    csc_model = None,
                    apply_prox = False,
                    current_batch_size = None,
                    src_task_ids = []):

        actor_loss_out = 0.0
        critic_loss_out = 0.0
        critic_prox_out = 0.0
        actor_prox_out = 0.0
        list_prox_coefs = [self.prox_coef]

        for it in range(iterations):

            if len(src_task_ids) > 0:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample_tasks(task_ids = src_task_ids, batch_size = current_batch_size)

            else:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(current_batch_size)

            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)


            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)


            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

            if csc_model is None:
                beta_score = torch.ones((current_batch_size, 1)).to(self.device)

            else:
                beta_score, clipping_factor = self.get_propensity(csc_model, pre_act_rew, obs)
                beta_score = beta_score.to(self.device)
                list_prox_coefs.append(self.prox_coef)

            noise = (torch.randn_like(action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)


            critic_loss_temp = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
            assert critic_loss_temp.shape == beta_score.shape, ('shape critic_loss_temp and beta_score shoudl be the same', critic_loss_temp.shape, beta_score.shape)

            critic_loss = (critic_loss_temp * beta_score).mean()
            critic_loss_out += critic_loss.item()

            if apply_prox:
                critic_prox = self.get_prox_penalty(self.critic, self.ckpt['critic'])
                critic_loss = critic_loss + self.prox_coef * critic_prox
                critic_prox_out += critic_prox.item()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % self.policy_freq == 0:

                actor_loss_temp = -1 * beta_score * self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew)
                actor_loss = actor_loss_temp.mean()
                actor_loss_out += actor_loss.item()

                if apply_prox:
                    actor_prox = self.get_prox_penalty(self.actor, self.ckpt['actor'])
                    actor_loss = actor_loss + self.prox_coef * actor_prox
                    actor_prox_out += actor_prox.item()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

        out = {}
        if iterations == 0:
            out['critic_loss'] = 0
            out['actor_loss']  = 0
            out['prox_critic'] = 0
            out['prox_actor']  = 0
            out['beta_score']  = 0

        else:
            out['critic_loss'] = critic_loss_out/iterations
            out['actor_loss']  = self.policy_freq * actor_loss_out/iterations
            out['prox_critic'] = critic_prox_out/iterations
            out['prox_actor']  = self.policy_freq * actor_prox_out/iterations
            out['beta_score']  = beta_score.cpu().data.numpy().mean()

        out['avg_prox_coef'] = np.mean(list_prox_coefs)

        return out

    def save(self, filename):
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'actor_target_state_dict': self.actor_target.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'prox_coef': self.prox_coef,
                    'prox_coef_init': self.prox_coef_init,
                    'lam_csc': self.lam_csc,
                    'use_ess_clipping': self.use_ess_clipping,
                    'use_normalized_beta': self.use_normalized_beta,
                    'enable_beta_obs_cxt': self.enable_beta_obs_cxt,
                    'batch_size': self.batch_size,
                    'max_action': self.max_action,
                    'policy_noise': self.policy_noise,
                    'noise_clip': self.noise_clip,
                    'gamma': self.gamma,
                    'policy_freq': self.policy_freq,
                    'ptau': self.ptau,
                    'device': self.device,
                    'max_iter_logistic': self.max_iter_logistic,
                    'beta_clip': self.beta_clip,
                    'type_of_training': self.type_of_training,
                    'reset_optims': self.reset_optims,
                    'train_tasks_list': self.train_tasks_list
                    }, filename)



    def train_TD3(
                self,
                replay_buffer=None,
                iterations=None,
                tasks_buffer = None,
                train_iter = 0,
                task_id = None,
                nums_snap_trains = 5):


        actor_loss_out = 0.0
        critic_loss_out = 0.0



        for it in range(iterations):


            x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(self.batch_size)
            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)

            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)

            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

            noise = (torch.randn_like(action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_loss_out += critic_loss.item()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

 
            if it % self.policy_freq == 0:

                actor_loss = -self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew).mean()
                actor_loss_out += actor_loss.item()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

        out = {}
        out['critic_loss'] = critic_loss_out/iterations
        out['actor_loss'] = self.policy_freq * actor_loss_out/iterations

        self.copy_model_params()
        return out, None
    
def adapt(self,
          train_replay_buffer=None,
          train_tasks_buffer=None,
          eval_task_buffer=None,
          task_id=None,
          snap_iter_nums=5,
          main_snap_iter_nums=15,
          sampling_style='replay',
          sample_mult=1):
    
    self.actor_optimizer = optim.Adam(self.actor.parameters())
    self.critic_optimizer = optim.Adam(self.critic.parameters())

    csc_model, csc_info = self.train_cs(task_id=task_id,
                                        snap_buffer=eval_task_buffer,
                                        train_tasks_buffer=train_tasks_buffer,
                                        adaptation_step=True)

    out_single = self.do_training(replay_buffer=eval_task_buffer.get_buffer(task_id),
                                  iterations=snap_iter_nums,
                                  csc_model=None,
                                  apply_prox=False,
                                  current_batch_size=eval_task_buffer.size_rb(task_id))

    out_single['csc_info'] = csc_info
    out_single['snap_iter'] = snap_iter_nums

    out = self.do_training(replay_buffer=train_replay_buffer,
                           iterations=main_snap_iter_nums,
                           csc_model=csc_model,
                           apply_prox=True,
                           current_batch_size=sample_mult * self.batch_size)

    return out, out_single

def rollback(self):
    self.actor.load_state_dict(self.actor_copy.state_dict())
    self.actor_target.load_state_dict(self.actor_target_copy.state_dict())
    self.critic.load_state_dict(self.critic_copy.state_dict())
    self.critic_target.load_state_dict(self.critic_target_copy.state_dict())
    self.actor_optimizer.load_state_dict(self.actor_optimizer_copy.state_dict())
    self.critic_optimizer.load_state_dict(self.critic_optimizer_copy.state_dict())

def save_model_states(self):
    self.actor_copy = deepcopy(self.actor)
    self.actor_target_copy = deepcopy(self.actor_target)
    self.critic_copy = deepcopy(self.critic)
    self.critic_target_copy = deepcopy(self.critic_target)
    self.actor_optimizer_copy = deepcopy(self.actor_optimizer)
    self.critic_optimizer_copy = deepcopy(self.critic_optimizer)

def set_training_style(self):
    self.training_func = self.train_TD3

def train(self,
          replay_buffer=None,
          iterations=None,
          tasks_buffer=None,
          train_iter=0,
          task_id=None,
          nums_snap_trains=5):

    return self.training_func(replay_buffer=replay_buffer,
                              iterations=iterations,
                              tasks_buffer=tasks_buffer,
                              train_iter=train_iter,
                              task_id=task_id,
                              nums_snap_trains=nums_snap_trains)





class MultiTasksSnapshot(object):
    def __init__(self, max_size=1e3):
        self.max_size = max_size
        self.task_buffers = {}

    def init(self, task_ids=None):
        self.task_buffers = {idx: Buffer(max_size=self.max_size) for idx in task_ids}

    def reset(self, task_id):
        self.task_buffers[task_id].reset()

    def list(self):
        return list(self.task_buffers.keys())

    def add(self, task_id, data):
        self.task_buffers[task_id].add(data)

    def size_rb(self, task_id):
        return self.task_buffers[task_id].size_rb()

    def get_buffer(self, task_id):
        return self.task_buffers[task_id]

    def sample(self, task_ids, batch_size):
        if len(task_ids) == 1:
            xx, _, _, _, _, pu, pr, px, _, _, _ = self.task_buffers[task_ids[0]].sample(batch_size)
            return pu, pr, px, xx

        mb_actions = []
        mb_rewards = []
        mb_obs = []
        mb_x = []

        for tid in task_ids:
            xx, _, _, _, _, pu, pr, px, _, _, _ = self.task_buffers[tid].sample(batch_size)
            mb_actions.append(pu)
            mb_rewards.append(pr)
            mb_obs.append(px)
            mb_x.append(xx)

        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_x = np.asarray(mb_x, dtype=np.float32)

        return mb_actions, mb_rewards, mb_obs, mb_x

    def sample_tasks(self, task_ids, batch_size):
        mb_xx = []
        mb_yy = []
        mb_u = []
        mb_r = []
        mb_d = []
        mb_pu = []
        mb_pr = []
        mb_px = []
        mb_nu = []
        mb_nr = []
        mb_nx = []
        shuffled_task_ids = random.sample(task_ids, len(task_ids))
        for tid in shuffled_task_ids:
            xx, yy, u, r, d, pu, pr, px, nu, nr, nx = self.task_buffers[tid].sample(batch_size)
            mb_xx.append(xx)
            mb_yy.append(yy)
            mb_u.append(u)
            mb_r.append(r)
            mb_d.append(d)
            mb_pu.append(pu)
            mb_pr.append(pr)
            mb_px.append(px)
            mb_nu.append(nu)
            mb_nr.append(nr)
            mb_nx.append(nx)
        mb_xx = np.asarray(mb_xx, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)
        mb_yy = np.asarray(mb_yy, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)
        mb_u = np.asarray(mb_u, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)
        mb_r = np.asarray(mb_r, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)
        mb_d = np.asarray(mb_d, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)
        mb_pu = np.asarray(mb_pu, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)
        mb_pr = np.asarray(mb_pr, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) 
        mb_px = np.asarray(mb_px, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) 
        mb_nu = np.asarray(mb_nu, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) 
        mb_nr = np.asarray(mb_nr, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) 
        mb_nx = np.asarray(mb_nx, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) 

        return mb_xx, mb_yy, mb_u, mb_r, mb_d, mb_pu, mb_pr, mb_px, mb_nu, mb_nr, mb_nx