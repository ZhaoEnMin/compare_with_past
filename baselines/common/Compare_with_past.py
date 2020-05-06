import numpy as np
import tensorflow as tf
import random
from gym import spaces
from baselines.a2c.utils import discount_with_dones,discount_with_dones_equal
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.a2c.policies import nature_cnn
from baselines.common.input import observation_input
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


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        
        self.var = np.ones(shape, 'float64')
        self.count = epsilon


    def update(self, x):

        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class compare_with_past(object):
    
    def __init__(self, fn_reward=np.sign, fn_obs=None,
            n_env=16, batch_size=128, n_update=4, 
            max_steps=int(1e5), gamma=0.99, stack=1):#fn_reward ==np.sign

        self.obsbuffer=RunningMeanStd()
        self.rewbuffer=RunningMeanStd()
        self.rff_int=RewardForwardFilter(gamma)
        self.buf_rews_train()

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
        obsave=[]
        rsave=[]

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
                actions.append(action)
                rewards.append(self.fn_reward(reward))
                dones.append(False)
        dones[len(dones)-1]=True
        returns = discount_with_dones_equal(rewards, dones)
        obsofrew=np.zeros((len(dones),1))
        for ik in range(int(len(dones)/2048)+1):
            if ik<int(len(dones)/2048):
                obsofrew[ik*2048:ik*2048+2048,0]=np.array(self.sess.run(self.int_rew,feed_dict={self.obss:np.array(obs[ik*2048:ik*2048+2048])}))
            else:
                obsofrew[ik*2048:len(dones),0]=np.array(self.sess.run(self.int_rew,feed_dict={self.obss:np.array(obs[ik*2048:len(dones)])}))
        
        obssortrew=np.sort(obsofrew,0)
        obla=obsofrew.shape[0]
        obmax025=obssortrew[int(0.75*obla)]
        i=0
        rmax=0
        for (ob, action, R) in list(zip(obs, actions, returns)):
            if obmax025<obsofrew[i] and R -rmax>0.1:
                self.buffer.add(ob,action, R)
            rmax=R
            i=i+1

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
        if len(self.buffer) > 0:
            obs,action,reward=self.buffer.sample(batch_size)
            return obs,reward.reshape((batch_size,1))
        else:
            return None,None


   
    def buf_rews_train(self):
        tf.reset_default_graph()
        self.Graph=tf.Graph()
        with self.Graph.as_default():
            self.obss=tf.placeholder(tf.float32,[None,84,84,4])
            self.rewards=tf.placeholder(tf.float32,[None,1])
            xinput = self.obss[:,:,:,-1:]
            xinput = tf.clip_by_value((xinput - self.obsbuffer.mean) / (self.obsbuffer.var**0.5), -5.0, 5.0)
            xr = tf.nn.leaky_relu(conv(xinput, 'c1r', nf=32 * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
            xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=32 * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
            xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=32 * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
            rgbr = conv_to_fc(xr)
            X_r = tf.nn.relu(fc(rgbr, 'fc1r', nh=512, init_scale=np.sqrt(2)))
            X_r = tf.nn.relu(fc(X_r, 'fc2r', nh=512, init_scale=np.sqrt(2)))
            
            X_r = fc(X_r, 'fc3r', nh=512, init_scale=np.sqrt(2))

            xrr = tf.nn.leaky_relu(conv(xinput, 'c1rr', nf=32 * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
            xrr = tf.nn.leaky_relu(conv(xrr, 'c2rr', nf=32 * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
            xrr = tf.nn.leaky_relu(conv(xrr, 'c3rr', nf=32 * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
            rgbrr = [conv_to_fc(xrr)]
            X_rr = fc(rgbrr[0], 'fc1rr', nh=512, init_scale=np.sqrt(2))

            self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_rr) - X_r),-1)
            targets = tf.stop_gradient(X_rr)
            self.aux_loss = tf.reduce_mean(tf.square(targets +tf.sqrt(self.rewards/512)- X_r),-1)
            #self.aux_loss=self.aux_loss-self.rewards
            #self.rew_trainloss =tf.reduce_mean(tf.square(self.aux_loss-self.rewards),-1)
            mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
            mask = tf.cast(mask < 1, tf.float32)
            self.rew_trainloss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
            opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
            self.train_op = opt.minimize(self.rew_trainloss)
            
            self.sess=tf.Session()
            self.sess.run(tf.global_variables_initializer())
    def cwp_train(self,obs):
        for i in range(4):
            self.sess.run(self.train_op,feed_dict={self.obss:obs[0:512,:,:,:],self.rewards:np.zeros((512,1))})
            self.sess.run(self.train_op,feed_dict={self.obss:obs[512:1024,:,:,:],self.rewards:np.zeros((512,1))})
            self.sess.run(self.train_op,feed_dict={self.obss:obs[1024:1536,:,:,:],self.rewards:np.zeros((512,1))})
            self.sess.run(self.train_op,feed_dict={self.obss:obs[1536:2048,:,:,:],self.rewards:np.zeros((512,1))})
        obstrue,rewardstrue= self.sample_batch(128)
        if obstrue is None:
            obstrue=None
        else:
            self.sess.run(self.train_op,feed_dict={self.obss:obstrue,self.rewards:rewardstrue})
        #return self.sess.run(self.rew_trainloss,feed_dict={self.obss:obs[0:512,:,:,:]})

    def cwp_getint(self,obs):
        return self.sess.run(self.int_rew,feed_dict={self.obss:obs})