#! /usr/bin/python3
import time
import threading
from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ReflexAgent import *
from cache.DataLoader import DataLoaderPintos

if __name__ == "__main__":
    # disk activities
    datafile = sys.argv[1]
    dataloader = DataLoaderPintos([datafile])
    
    #sizes = [1000, 2000, 3000, 4000]
    #sizes = [5, 25, 50, 100, 300]
    #sizes = [50, 100, 200, 400]
    sizes = [10000, 50000, 100000, 200000]
    for cache_size in sizes:
        
        print("==================== Cache Size: %d ====================" % cache_size)

        # cache
        env = Cache(dataloader, cache_size
            , feature_selection=('Base',)
            , reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
            , allow_skip=False
        )
        
        # agents
        agents = {}
        num_of_clusters = 8
        agents['DQN'] = DQNAgent(num_of_clusters, num_of_clusters*3,
        #agents['DQN'] = DQNAgent(env.n_actions, env.n_actions*3,
            learning_rate=0.01,
            reward_decay=0.9,

            # Epsilon greedy
            e_greedy_min=(0.0, 0.1),
            e_greedy_max=(0.2, 0.8),
            e_greedy_init=(0.1, 0.5),
            e_greedy_increment=(0.005, 0.01),
            e_greedy_decrement=(0.005, 0.001),

            history_size=50,
            dynamic_e_greedy_iter=25,
            reward_threshold=3,
            explore_mentor = 'LRU',

            replace_target_iter=100,
            memory_size=10000,
            batch_size=128,

            output_graph=False,
            verbose=0
        )
        #agents['Random'] = RandomAgent(env.n_actions)
        #agents['LRU'] = LRUAgent(env.n_actions)
        #agents['LFU'] = LFUAgent(env.n_actions)
        #agents['MRU'] = MRUAgent(env.n_actions)
        for (name, agent) in agents.items():

            print("-------------------- %s --------------------" % name)
            step = 0
            miss_rates = []    # record miss rate for every episode
            # determine how many episodes to proceed
            # 100 for learning agents, 20 for random agents
            # 1 for other agents because their miss rates are invariant
            if isinstance(agent, LearnerAgent):
                episodes = 100
            elif isinstance(agent, RandomAgent):
                episodes = 20
            else:
                episodes = 1

            for episode in range(episodes):
                # initial observation
                begin = time.time()
                observation = env.reset()

                while True:
                    # agent choose action based on observation
                    #action = agent.choose_action(observation)
                    action = agent.choose_action_new(observation)

                    # agent take action and get next observation and reward
                    observation_, reward = env.step(action)

                    # break while loop when end of this episode
                    if env.hasDone():
                        break

                    agent.store_transition(observation, action, reward, observation_)

                    if isinstance(agent, LearnerAgent) and (step > 20) and (step % 5 == 0):
                        agent.learn()

                    # swap observation
                    observation = observation_

                    if step % 100 == 0:
                        mr = env.miss_rate()
                        print("Agent=%s, Size=%d, Step=%d, Accesses=%d, Misses=%d, MissRate=%f"
                          % (name, cache_size, step, env.total_count, env.miss_count, mr)
                        )
                    step += 1

                # report after every episode
                end = time.time()
                mr = env.miss_rate()
                print("Agent=%s, Size=%d, Episode=%d: Accesses=%d, Misses=%d, MissRate=%f, Duration=%f"
                    % (name, cache_size, episode, env.total_count, env.miss_count, mr, end-begin)
                )
                miss_rates.append(mr)
            
            miss_rates = np.array(miss_rates)
            print("Agent=%s, Size=%d: Mean=%f, Median=%f, Max=%f, Min=%f"
                % (name, cache_size, np.mean(miss_rates), np.median(miss_rates), np.max(miss_rates), np.min(miss_rates))
            )
            # save model
            if isinstance(agent, LearnerAgent):
                agent.save(cache_size, datafile)
