import numpy as np
import tensorflow as tf
import random
from gym import spaces
from baselines.a2c.utils import discount_with_dones,discount_with_dones_equal
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.a2c.policies import nature_cnn
class ReplayBuffer(object):
    def __init__(self, size):
        
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, obs_t, action, R):#add replay buffer
        data = (obs_t, action, R)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, returns= [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            returns.append(R)
        return np.array(obses_t), np.array(actions), np.array(returns)

    def sample(self, batch_size):
        
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class compare_with_past(object):
    
    def __init__(self, fn_reward=np.sign, fn_obs=None,
            n_env=16, batch_size=512, n_update=4, 
            max_steps=int(1e5), gamma=0.99, stack=1):#fn_reward ==np.sign

        self.fn_reward = fn_reward
        self.fn_obs = fn_obs
        self.net={}
        self.buffer = ReplayBuffer(max_steps)
        self.buffer_negative = ReplayBuffer(max_steps)
        self.n_env = n_env
        self.batch_size = batch_size
        self.n_update = n_update

        self.max_steps = max_steps
        self.gamma = gamma
        self.stack = stack
        self.train_count = 0
        self.update_count = 0
        self.total_steps = []
        self.total_steps_negative = []
        self.total_rewards = []
        self.running_episodes = [[] for _ in range(n_env)]
        #self.buf_rews_int(sample_batch(self, batch_size)[0],sample_batch(self, batch_size)[1]):
        self.buf_rews_train()

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
        dones = []

        if self.stack > 1:
            ob_shape = list(trajectory[0][0].shape)
            nc = ob_shape[-1]
            ob_shape[-1] = nc*self.stack
            stacked_ob = np.zeros(ob_shape, dtype=trajectory[0][0].dtype)
        for (ob, action, reward) in trajectory:
            if ob is not None:
                x = self.fn_obs(ob) if self.fn_obs is not None else ob
                if self.stack > 1:
                    stacked_ob = np.roll(stacked_ob, shift=-nc, axis=2)
                    stacked_ob[:, :, -nc:] = x
                    obs.append(stacked_ob)
                else:
                    obs.append(x)
            else:
                obs.append(None)
            actions.append(action)
            rewards.append(self.fn_reward(reward))
            dones.append(False)
        dones[len(dones)-1]=True
        returns = discount_with_dones_equal(rewards, dones)
        for (ob, action, R) in list(zip(obs, actions, returns)):
            self.buffer.add(ob, action, R)

    def add_episode_negative(self, trajectory):
        obs = []
        actions = []
        rewards = []
        dones = []

        if self.stack > 1:
            ob_shape = list(trajectory[0][0].shape)
            nc = ob_shape[-1]
            ob_shape[-1] = nc*self.stack
            stacked_ob = np.zeros(ob_shape, dtype=trajectory[0][0].dtype)
        for (ob, action, reward) in trajectory:
            if ob is not None:
                x = self.fn_obs(ob) if self.fn_obs is not None else ob
                if self.stack > 1:
                    stacked_ob = np.roll(stacked_ob, shift=-nc, axis=2)
                    stacked_ob[:, :, -nc:] = x
                    obs.append(stacked_ob)
                else:
                    obs.append(x)
            else:
                obs.append(None)
            actions.append(action)
            rewards.append(0)#negative ==0
            dones.append(False)
        dones[len(dones)-1]=True
        #returns = discount_with_dones(rewards, dones, self.gamma)
        for (ob, action, R) in list(zip(obs, actions, returns)):
            self.buffer_negative.add(ob, action, R)

    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.max_steps and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)
        else:
            self.add_episode_negative(trajectory)
            self.total_steps_negative.append(trajectory)
            while np.sum(self.total_steps_negative)>self.max_steps and len(self.total_steps_negative)>1:
                self.total_steps_negative.pop(0)



    def num_steps(self):
        return len(self.buffer)

    def num_episodes(self):
        return len(self.total_rewards)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def step(self, obs, actions, rewards, dones):
        for n in range(self.n_env):
            if self.n_update > 0:
                self.running_episodes[n].append([obs[n], actions[n], rewards[n]])
            else:
                self.running_episodes[n].append([None, actions[n], rewards[n]])

        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []

    def sample_batch(self, batch_size):
        if len(self.buffer) > batch_size/2 and len(self.buffer_negative)>batch_size/2:
            batch_size = min(batch_size, len(self.buffer))
            batch_size_nega = min(batch_size, len(self.buffer_negative))
            obs,action,reward=self.buffer.sample(batch_size)
            obs_nega,action_nega,reward_nega=self.buffer_negative.sample(batch_size_nega)
            return np.concatenate([obs,obs_nega]),np.concatenate([reward,reward_nega])
        elif len(self.buffer_negative)>batch_size and len(self.buffer)<batch_size/2:
            obs_nega,action_nega,reward_nega=self.buffer_negative.sample(batch_size)
            return obs_nega,reward_nega
        else:
            return None,None
 
    def cwp_train(self):
        return self.sample_batch(self.batch_size)


   