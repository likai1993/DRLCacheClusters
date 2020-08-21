import sys, time, os, random
import numpy as np
from sklearn.cluster import KMeans
from cache.DataLoader import *
from cache.IOSimulator import miss
from collections import Counter # improve the counter speed in _elapsed_requests
from queue import Queue 

class Cache(object):
    def __init__(self, requests, cache_size
        # Time span for different terms
        #, terms=[10, 100, 1000]
        #, terms=[100, 1000, 10000]
        , terms=[100, 500, 1000, 3000, 5000, 10000]

        # Features. Available 'Base', 'UT', and 'CT'
        # Refer to the report for what they mean
        , feature_selection=('Base',)

        # Reward functions. Here are the default params
        # 1. Zhong et. al. - (name='zhong', short_reward=1.0, long_span=100, beta=0.5)
        # 2. Ours - (name='our', alpha=0.5, psi=10, mu=1, beta=0.3), let psi=0 to remove penalty
        , reward_params=dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
                 
        # leave none for random read/writes
        , operations=None
        , allow_skip=False
        , cluster_num=8
    ):
        # If the cache allows skip eviction
        # Network caching, disk caching do allow skipping eviction and grab resource directly.
        # However, cache between CPU and memory requires every accessed data to be cached beforehand.
        self.allow_skip = allow_skip
        self.cluster_num = cluster_num 

        # Counters
        self.total_count = 0
        self.miss_count = 0
        self.evict_count = 0

        # Load requests
        if isinstance(requests, DataLoader):   # From data loader
            self.requests = requests.get_requests()
            self.operations = requests.get_operations()
            self.tableIDs = requests.get_tableIDs()
        else:                                   # From array
            self.requests = requests
            self.operations = operations
            # random read/writes
            if self.operations is None:
                self.operations = [random.randint(0, 1) for i in range(len(self.requests))]
        self.cur_index = -1

        if len(self.requests) <= cache_size:
            raise ValueError("The count of requests are too small. Try larger one.")

        if len(self.requests) != len(self.operations):
            raise ValueError("Not every request is assigned with an operation.")
        
        # Important: Reward function
        self.reward_params = reward_params
        
        # Elasped terms - short, middle and long
        self.FEAT_TREMS = terms

        # Cache
        self.cache_size = cache_size
        self.slots = [-1] * self.cache_size
        self.access_bits = [False] * self.cache_size
        self.dirty_bits = [False] * self.cache_size

        # page statistics
        self.queue1 = Queue(maxsize = terms[0])
        self.queue2 = Queue(maxsize = terms[1])
        self.queue3 = Queue(maxsize = terms[2])
        self.queue4 = Queue(maxsize = terms[3])
        self.queue5 = Queue(maxsize = terms[4])
        self.queue6 = Queue(maxsize = terms[5])

        self.page_features = [0,0,0,0,0,0] * self.cache_size
        self.page_features_table = [0] * self.cache_size
        self.cluster_features_sum = [0,0,0,0,0,0] * (self.cluster_num)
        self.check_miss_rate = False


        # Action & feature information
        self.sel_features = feature_selection
        self.n_actions = self.cache_size + 1 if allow_skip else self.cache_size
        self.n_features = 0
        if 'Base' in self.sel_features:
            self.n_features += (self.cache_size + 1) * len(self.FEAT_TREMS)
        if 'UT' in self.sel_features:
            self.n_features += self.cache_size
        if 'CT' in self.sel_features:
            self.n_features += self.cache_size

    # Display the current cache state
    def display(self):
        print(self.slots)

    # Return miss rate
    def miss_rate(self):
        return self.miss_count / self.total_count

    def reset(self):
        self.total_count = 0
        self.miss_count = 0

        self.cur_index = 0

        self.slots = [-1] * self.cache_size
        self.access_bits = [False] * self.cache_size
        self.dirty_bits = [False] * self.cache_size

        slot_id = 0
        while slot_id < self.cache_size and self.cur_index < len(self.requests):
            request = self._current_request()
            if request not in self.slots:
                self.miss_count += 1
                self.slots[slot_id] = request
                self.page_features_table[slot_id] = self.tableIDs[self.cur_index]
                self._hit_cache(slot_id)
                self.update_queues(slot_id)
                slot_id += 1
            self.total_count += 1
            self.cur_index += 1

        # Back to the last requested index
        self.cur_index -= 1

        # Run to the first miss as the inital decision epoch.
        self._run_until_miss()

        return self._get_observation()

    # Has program finished?
    def hasDone(self):
        return self.cur_index == len(self.requests)

    # Make action at the current decision epoch and run to the
    # next decision epoch.
    def step(self, action, training=True, clustering=True):
        if self.hasDone():
            raise ValueError("Simulation has finished, use reset() to restart simulation.")

        candidates = np.where(self.clusters.labels_==action)[0]
        action = candidates[random.randrange(0, len(candidates))]

        if not self.allow_skip:
            action += 1

        if action < 0 or action > len(self.slots):
            raise ValueError("Invalid action %d taken." % action)

        # Evict slot of (aciton - 1). action == 0 means skipping eviction.
        if action != 0:
            out_resource = self.slots[action - 1]
            in_resource = self._current_request()
            slot_id = action - 1
            self.slots[slot_id] = in_resource
            self.page_features_table[slot_id] = self.tableIDs[self.cur_index]
            self._hit_cache(slot_id)
            self.evict_count += 1
            # update queues and page statistic
            self.update_queues(slot_id)
        else:
            skip_resource = self._current_request()

        last_index = self.cur_index

        # Proceed kernel and resource accesses until next miss.
        self._run_until_miss(training)

        # Get observation.
        observation = self._get_observation(clustering)

        # Zhong et. al. 2018
        reward=0.0
        if training:
            if self.reward_params['name'].lower() == "zhong":
                # Compute reward: R = short term + long term
                reward = 0.0

                # Total count of hit since last decision epoch
                hit_count = self.cur_index - last_index - 1
                if hit_count != 0: reward += self.reward_params['short_reward']
                # Long term
                start = last_index
                end = last_index + self.reward_params['long_span']
                if end > len(self.requests): end = len(self.requests)
                long_term_hit = 0
                next_reqs = self.requests[start : end]
                for rc in self.slots:
                    long_term_hit += next_reqs.count(rc)
                reward += self.reward_params['beta'] * long_term_hit / (end - start)

            # Ours
            elif self.reward_params['name'].lower() == "our":
                # Compute reward: R = hit reward + miss penalty
                reward = 0.0

                hit_count = self.cur_index - last_index - 1
                reward += hit_count

                miss_resource = self._current_request()
                # If evction happens at last decision epoch
                if action != 0:
                    # Compute the swap-in reward
                    past_requests = self.requests[last_index + 1 : self.cur_index]
                    reward += self.reward_params['alpha'] * past_requests.count(in_resource)
                    # Compute the swap-out penalty
                    if miss_resource == out_resource:
                        reward -= self.reward_params['psi'] / (hit_count + self.reward_params['mu'])
                # Else no evction happens at last decision epoch 
                else:
                    # Compute the reward of skipping eviction
                    reward += self.reward_params['beta'] * reward
                    # Compute the penalty of skipping eviction
                    if miss_resource == skip_resource:
                        reward -= self.reward_params['psi'] / (hit_count + self.reward_params['mu'])

        return observation, reward

    # update s/m/l queues and page statistic
    def update_queue(self, queue, index, slot_id):
        if queue.full():
            f_slot_id = queue.get()
            if hasattr(self, "clusters"):
                self.cluster_features_sum[self.clusters.labels_[f_slot_id]*6 + index] -= 1 
            else:
                self.page_features[f_slot_id*6+index] -= 1
        queue.put(slot_id)
        if hasattr(self, "clusters"):
            self.cluster_features_sum[self.clusters.labels_[slot_id]*6+index] += 1 
        else:
            self.page_features[slot_id*6+index] += 1


    def update_queues(self, slot_id):
        self.update_queue(self.queue1, 0, slot_id)
        self.update_queue(self.queue2, 1, slot_id)
        self.update_queue(self.queue3, 2, slot_id)
        self.update_queue(self.queue4, 3, slot_id)
        self.update_queue(self.queue5, 4, slot_id)
        self.update_queue(self.queue6, 5, slot_id)

    # Run until next cache miss
    def _run_until_miss(self, training=True):
        self.cur_index += 1
        while self.cur_index < len(self.requests):
            request = self._current_request()
            self.total_count += 1
            if request not in self.slots:
                self.miss_count += 1
                if not training:
                    miss()
                break
            else:
                slot_id = self.slots.index(request)
                self._hit_cache(slot_id)
                # update queues
                self.update_queues(slot_id)

            self.cur_index += 1
        return self.hasDone()

    # In case that the simulation has ended, but we still need the current
    # request for the last observation and reward. Return -1 to eliminate
    # any defects.
    def _current_request(self):
        return -1 if self.hasDone() else self.requests[self.cur_index]

    # Simulate cache hit, update attributes.
    def _hit_cache(self, slot_id):
        # Set access bit
        self.access_bits[slot_id] = True
        # If the operation is write
        if self.operations[self.cur_index] == 1:
            # Set dirty bit
            self.dirty_bits[slot_id] = True

    # Return page features 
    def _get_features(self):
        # [Freq, F1, F2, ..., Fc] where Fi = [Rs, Rm, Rl]
        # i.e. the request times in short/middle/long term for each
        features = np.array(self.page_features)
        return features

    def notify_miss_rate(self):
        self.check_miss_rate = True

    # Re-clustering by monitoring miss rate
    def refresh_clusters(self, threshold):
        last_miss_rate = 0
        consecutive_drop_times = 0
        while (True):
            if self.check_miss_rate:
                cur_miss_rate = self.miss_rate()
                #print("check miss rate...", cur_miss_rate, consecutive_drop_times)
                if cur_miss_rate > last_miss_rate:
                    consecutive_drop_times += 1
                else:
                    if consecutive_drop_times >= 1:
                        consecutive_drop_times -= 1

                if consecutive_drop_times >= threshold:
                    print("re-clustering...")
                    observation=dict(features=self._get_features())
                    self._do_cluster(observation)
                    consecutive_drop_times = 0

                # wait for next notification
                last_miss_rate = cur_miss_rate 
                self.check_miss_rate = False
            time.sleep(2)

    #return cluster_features
    def get_cluster_features(self):
        cluster_features = np.array([])
        # calculate centroid of each cluster
        for label in range(np.max(self.clusters.labels_)+1):
            indices = np.where(self.clusters.labels_==label)[0]
            cluster_features = np.concatenate([cluster_features, np.array([self.cluster_features_sum[label*3]/len(indices), self.cluster_features_sum[label*3+1]/len(indices), self.cluster_features_sum[label*3+2]/len(indices),self.cluster_features_sum[label*3+3]/len(indices) , self.cluster_features_sum[label*3+4]/len(indices), self.cluster_features_sum[label*3+5]/len(indices)])], axis=0)
        return cluster_features

    def _do_cluster(self, observation):
        begin = time.time()
        num_rows = len(observation['features'])/len(self.FEAT_TREMS)
        page_features = observation['features'].reshape(int(num_rows), len(self.FEAT_TREMS))
        page_tables = np.array(self.page_features_table).reshape(int(num_rows), 1)
        page_features = np.concatenate((page_features, page_tables), 1)
        self.clusters = KMeans(n_clusters=self.cluster_num, init="random", n_init=1, random_state=0).fit(page_features)
        #print("clustering time:", time.time()-begin)
        # update cluster_features
        self.cluster_features_sum = [0]*len(self.FEAT_TREMS) * (self.cluster_num)
        for i in range(self.cache_size):
            self.cluster_features_sum[self.clusters.labels_[i]*3] += self.page_features[i*3]
            self.cluster_features_sum[self.clusters.labels_[i]*3+1] += self.page_features[i*3+1]
            self.cluster_features_sum[self.clusters.labels_[i]*3+2] += self.page_features[i*3+2]
            self.cluster_features_sum[self.clusters.labels_[i]*3+3] += self.page_features[i*3+3]
            self.cluster_features_sum[self.clusters.labels_[i]*3+4] += self.page_features[i*3+4]
            self.cluster_features_sum[self.clusters.labels_[i]*3+5] += self.page_features[i*3+5]

    def _get_observation_old_clusters(self):
        cluster_features= dict(features=self.get_cluster_features())
        return cluster_features

    def _get_observation(self, clustering = True):
        if clustering:
            observation=dict(features=self._get_features())   
            # keep re-clustering
            self._do_cluster(observation)
            # do not pass the 4-th page features (table related) to DQN
            #features = np.concatenate([self.clusters.cluster_centers_[i] for i in range(np.max(self.clusters.labels_)+1)])
            features = np.concatenate([self.clusters.cluster_centers_[i][0:-1] for i in range(np.max(self.clusters.labels_)+1)])
            cluster_features= dict(features=features)
            return cluster_features
        else:
            # keep using old clusters
            return self._get_observation_old_clusters() 
