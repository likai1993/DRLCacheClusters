"""
This code is partially from Morvan Zhou
https://morvanzhou.github.io/tutorials/

We add neccessary decision procedures for our cache policy.

Using:
Tensorflow: 1.0
"""
import os.path
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from agents.CacheAgent import LearnerAgent
from agents.ReflexAgent import RandomAgent, LRUAgent, LFUAgent

np.random.seed(1)
tf.set_random_seed(1)

# disable eager execution
tf.disable_eager_execution()

# Deep Q Network
class DQNAgent(LearnerAgent):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        
        e_greedy_min=(0.1, 0.1),
        e_greedy_max=(0.1, 0.1),

        # leave either e_greedy_init or e_greedy_decrement None to disable epsilon greedy
        # only leave e_greedy_increment to disable dynamic bidirectional epsilon greedy
        e_greedy_init=None,
        e_greedy_increment=None,
        e_greedy_decrement=None,

        reward_threshold=None,
        history_size=10,
        dynamic_e_greedy_iter=5,
        explore_mentor = 'LRU',
        
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,

        output_graph=False,
        verbose=0
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size

        self.epsilons_min = e_greedy_min
        self.epsilons_max = e_greedy_max
        self.epsilons_increment = e_greedy_increment
        self.epsilons_decrement = e_greedy_decrement
        
        self.epsilons = list(e_greedy_init)
        if (e_greedy_init is None) or (e_greedy_decrement is None):
            self.epsilons = list(self.epsilons_min)

        self.explore_mentor = None
        if explore_mentor.upper() == 'LRU':
            self.explore_mentor = LRUAgent
        elif explore_mentor.upper() == 'LFU':
            self.explore_mentor = LFUAgent
        
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0
        
        # initialize a history set for rewards
        self.reward_history = []
        self.history_size = history_size
        self.dynamic_e_greedy_iter = dynamic_e_greedy_iter
        self.reward_threshold = reward_threshold

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        
        self.verbose = verbose

    def _build_net(self):
        # neccessary cleaning
        tf.reset_default_graph()
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, 32, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer. collections is used later when assign to target net
            with tf.variable_scope('32'):
                w3 = tf.get_variable('w2', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, s_):
        s, s_ = s['features'], s_['features']
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        
        # Record reward
        if len(self.reward_history) == self.history_size:
            self.reward_history.pop(0)
        self.reward_history.append(r)

    def choose_action(self, observation):
        # draw probability sample
        coin = np.random.uniform()
        if coin < self.epsilons[0]:
            action = RandomAgent._choose_action(self.n_actions)
        elif self.epsilons[0] <= coin and coin < self.epsilons[0] + self.epsilons[1]:
            action = self.explore_mentor._choose_action(observation)
        else:
            observation = observation['features']
            # to have batch dimension when feed into tf placeholder
            observation = observation[np.newaxis, :]

            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            
        if action < 0 or action > self.n_actions:
            raise ValueError("DQNAgent: Error index %d" % action)
        return action

    '''
    Added by Kai Li, to reduce the inference overhead by clustering
    '''
    def choose_action_new(self, observation):
        #print("features:", observation['features'])

        # draw probability sample
        coin = np.random.uniform()
        if coin < self.epsilons[0]:
            cluster_id = RandomAgent._choose_action(self.n_actions)
            #print("random selected cluster id", cluster_id)
        #elif self.epsilons[0] <= coin and coin < self.epsilons[0] + self.epsilons[1]:
            #cluster_id = self.explore_mentor._choose_action(cluster_observations)
        else:
            cluster_features = observation['features']
            # to have batch dimension when feed into tf placeholder
            cluster_features = cluster_features[np.newaxis, :]

            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: cluster_features})
            #print("actions_value", actions_value)
            cluster_id = np.argmax(actions_value)
            
        if cluster_id < 0 or cluster_id > self.n_actions:
            raise ValueError("DQNAgent: Error index %d" % cluster_id)

        return cluster_id

    def save(self, size):
        saver = tf.train.Saver()
        file_prefix = "DRLCache_model_1_"+str(size)
        model_file = "/tmp/"+file_prefix + "/" + file_prefix+ ".ckpt"
        save_path = saver.save(self.sess, model_file)
        print("Model saved in path: %s" % save_path)

    def load(self, size):
        saver = tf.train.Saver()
        file_prefix = "DRLCache_model_1_"+str(size)
        model_dir = "/tmp/"+file_prefix
        model_file = model_dir + "/" + file_prefix+ ".ckpt"
        if os.path.isdir(model_dir):
            saver.restore(self.sess, model_file)
            print("Model restored.")
            return True
        else:
            print("Model not exists.")
            return False

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # verbose
            if self.verbose >= 1:
                print('Target DQN params replaced')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
            feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target}
        )
        self.cost_his.append(self.cost)
        # verbose                    
        if (self.verbose == 2 and self.learn_step_counter % 100 == 0) or \
            (self.verbose >= 3 and self.learn_step_counter % 20 == 0):
            print("Step=%d: Cost=%d" % (self.learn_step_counter, self.cost))

        # increasing or decreasing epsilons
        if self.learn_step_counter % self.dynamic_e_greedy_iter == 0:

            # if we have e-greedy?
            if self.epsilons_decrement is not None:
                # dynamic bidirectional e-greedy
                if self.epsilons_increment is not None:
                    rho = np.median(np.array(self.reward_history))
                    if rho >= self.reward_threshold:
                        self.epsilons[0] -= self.epsilons_decrement[0]
                        self.epsilons[1] -= self.epsilons_decrement[1]
                        # verbose
                        if self.verbose >= 3:
                            print("Eps down: rho=%f, e1=%d, e2=%f" % (rho, self.epsilons[0], self.epsilons[1]))
                    else:
                        self.epsilons[0] += self.epsilons_increment[0]
                        self.epsilons[1] += self.epsilons_increment[1]
                        # verbose                    
                        if self.verbose >= 3:
                            print("Eps up: rho=%f, e1=%d, e2=%f" % (rho, self.epsilons[0], self.epsilons[1]))
                # traditional e-greedy
                else:
                    self.epsilons[0] -= self.epsilons_decrement[0]
                    self.epsilons[1] -= self.epsilons_decrement[1]

            # enforce upper bound and lower bound
            truncate = lambda x, lower, upper: min(max(x, lower), upper)
            self.epsilons[0] = truncate(self.epsilons[0], self.epsilons_min[0], self.epsilons_max[0])
            self.epsilons[1] = truncate(self.epsilons[1], self.epsilons_min[1], self.epsilons_max[1])

        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
