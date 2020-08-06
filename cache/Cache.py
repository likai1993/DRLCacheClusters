import sys, time, os, random
import numpy as np
from sklearn.cluster import KMeans
from cache.DataLoader import *
from collections import Counter # improve the counter speed in _elapsed_requests
                    
class Cache(object):
    def __init__(self, requests, cache_size
        # Time span for different terms
        #, terms=[10, 100, 1000]
        , terms=[1000, 10000, 100000]

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
    ):
        # If the cache allows skip eviction
        # Network caching, disk caching do allow skipping eviction and grab resource directly.
        # However, cache between CPU and memory requires every accessed data to be cached beforehand.
        self.allow_skip = allow_skip
        self.cluster_num = 8

        # Counters
        self.total_count = 0
        self.miss_count = 0
        self.evict_count = 0

        # Load requests
        if isinstance(requests, DataLoader):   # From data loader
            self.requests = requests.get_requests()
            self.operations = requests.get_operations()
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
        self.used_times = [-1] * self.cache_size
        self.cached_times = [-1] * self.cache_size
        self.access_bits = [False] * self.cache_size
        self.dirty_bits = [False] * self.cache_size
        self.resource_freq = {}

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
        # ... we've removed CS feature

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
        self.used_times = [-1] * self.cache_size
        self.access_bits = [False] * self.cache_size
        self.dirty_bits = [False] * self.cache_size

        slot_id = 0
        while slot_id < self.cache_size and self.cur_index < len(self.requests):
            request = self._current_request()
            if request not in self.slots:
                self.miss_count += 1
                self.slots[slot_id] = request
                self.cached_times[slot_id] = self.cur_index
                self._hit_cache(slot_id)
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


    '''
    Modified by Kai Li, select one from the cluster
    '''
    # Make action at the current decision epoch and run to the
    # next decision epoch.
    def step(self, action, is_training=True):
        if self.hasDone():
            raise ValueError("Simulation has finished, use reset() to restart simulation.")

        candidates = np.where(self.clusters.labels_==action)[0]
        action = candidates[random.randrange(0, len(candidates))]
        #print("victim:", action)
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
            self.cached_times[slot_id] = self.cur_index
            self._hit_cache(slot_id)
            self.evict_count += 1
        else:
            skip_resource = self._current_request()

        last_index = self.cur_index

        # Proceed kernel and resource accesses until next miss.
        self._run_until_miss()

        # Get observation.
        observation = self._get_observation(is_training)

        # Zhong et. al. 2018
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

    # Run until next cache miss
    def _run_until_miss(self):
        self.cur_index += 1
        while self.cur_index < len(self.requests):
            request = self._current_request()
            if request not in self.resource_freq:
                self.resource_freq[request] = 0
            self.resource_freq[request] += 1
            self.total_count += 1
            
            if request not in self.slots:
                self.miss_count += 1
                break
            else:
                slot_id = self.slots.index(request)
                self._hit_cache(slot_id)
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
        # Record last used time
        self.used_times[slot_id] = self.cur_index

    # The number of requests on rc_id among last `term` requests.
    '''
    def _elapsed_requests(self, term, rc_id):
        start = self.cur_index - term + 1
        if start < 0: start = 0
        end = self.cur_index + 1
        if end > len(self.requests): end = len(self.requests)
        #return self.requests[start : end].count(rc_id)
        return Counter(self.requests[start : end])[rc_id]
   
    '''
    # KAI modified to improve Count performance
    def _elapsed_requests(self, term):
        start = self.cur_index - term + 1
        if start < 0: start = 0
        end = self.cur_index + 1
        if end > len(self.requests): end = len(self.requests)
        C = Counter(self.requests[start : end])
        return np.array([C[rc] for rc in self.slots])

    # The number of requests on rc_id among next `term` requests.
    def _next_requests(self, term, rc_id):
        start = self.cur_index + 1
        if start < 0: start = 0
        end = self.cur_index + term
        if end > len(self.requests): end = len(self.requests)
        return self.requests[start : end].count(rc_id)

    # Return the observation features for reinforcement agent
    def _get_features(self):
        # [Freq, F1, F2, ..., Fc] where Fi = [Rs, Rm, Rl]
        # i.e. the request times in short/middle/long term for each
        # cached resource and the currently requested resource.

        # base
        #features = np.concatenate([
        #    np.array([self._elapsed_requests(t, self._current_request()) for t in self.FEAT_TREMS])
        #    , np.array([self._elapsed_requests(t, rc) for rc in self.slots for t in self.FEAT_TREMS])
        #], axis=0)

        features = np.concatenate([
            self._elapsed_requests(t) for t in self.FEAT_TREMS]
        )

        # last accessed time
        if 'UT' in self.sel_features:
            features = np.concatenate([
                features
                , np.array([self.used_times[i] for i in range(self.cache_size)])
            ], axis=0)
        # cached time
        if 'CT' in self.sel_features:
            features = np.concatenate([
                features
                , np.array([self.cached_times[i] for i in range(self.cache_size)])
            ], axis=0)
        
        return features

    '''
    Modified by Kai Li, to reduce the inference overhead by clustering
    '''

    def refresh_clusters(self, interval):
        while (True):
            time.sleep(interval)
            observation=dict(features=self._get_features())
            print("re-clustering...")
            self._do_cluster(observation)

    def get_cluster_features(self, page_features):
        # average access frequency
        cluster_features = np.array([]) 
        for label in range(np.max(self.clusters.labels_)+1):
            indices = np.where(self.clusters.labels_==label)[0]
            elements = np.array([page_features[i] for i in indices])
            # calculate new center
            cluster_features = np.concatenate([cluster_features, np.mean(elements, axis=0)], axis=0)
        return cluster_features

    def _do_cluster(self, observation):
        num_rows = len(observation['features'])/3
        page_features = observation['features'].reshape(int(num_rows), 3)
        self.clusters = KMeans(n_clusters=self.cluster_num, random_state=0).fit(page_features)

    def _get_observation_old_clusters(self):
        observation=dict(features=self._get_features(),
            cache_state=self.slots.copy(),
            cached_times=self.cached_times.copy(),
            last_used_times=self.used_times.copy(),
            total_use_frequency=[self.resource_freq.get(r, 0) for r in self.slots],
            access_bits=self.access_bits.copy(),
            dirty_bits=self.dirty_bits.copy()
        )
        num_rows = len(observation['features'])/3
        page_features = observation['features'].reshape(int(num_rows), 3)
        cluster_features= dict(features=self.get_cluster_features(page_features))
        return cluster_features

    def _get_observation(self, training = True):
        observation=dict(features=self._get_features(),
            cache_state=self.slots.copy(),
            cached_times=self.cached_times.copy(),
            last_used_times=self.used_times.copy(),
            total_use_frequency=[self.resource_freq.get(r, 0) for r in self.slots],
            access_bits=self.access_bits.copy(),
            dirty_bits=self.dirty_bits.copy()
        )
        if training:
            # keep reclustering
            self._do_cluster(observation)
            features = np.concatenate([self.clusters.cluster_centers_[i] for i in range(np.max(self.clusters.labels_)+1)])
            cluster_features= dict(features=features)
            return cluster_features
        else:
            # use old clusters' features
            return self._get_observation_old_clusters() 
