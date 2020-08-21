#! /usr/bin/python3
import time
import threading
from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ReflexAgent import *
from cache.DataLoader import * 

if __name__ == "__main__":
    # disk activities
    datafile1 = sys.argv[1]
    datafile2 = sys.argv[2]
    number_of_peroids = int(sys.argv[3])
    dataloader = DataLoaderMix([datafile1, datafile2], number_of_peroids)

    sizes = [10000, 50000, 100000, 200000]
    sizes = [7500]
    for cache_size in sizes:
        
        print("==================== Cache Size: %d ====================" % cache_size)

        # cache
        num_of_clusters = 5
        env = Cache(dataloader, cache_size
            , feature_selection=('Base',)
            , reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
            , allow_skip=False, cluster_num = num_of_clusters
        )
        
        # agents
        agents = {}
        agents['DQN'] = DQNAgent(num_of_clusters, num_of_clusters*6,
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
        for (name, agent) in agents.items():

            print("-------------------- %s --------------------" % name)
            step = 0
            miss_rates = []    # record miss rate for every episode
            # determine how many episodes to proceed
            # 100 for learning agents, 20 for random agents
            # 1 for other agents because their miss rates are invariant
            if isinstance(agent, LearnerAgent):
                episodes = 3 
                agent.load(cache_size, [datafile1, datafile2])
                # start parallel reclustering thread, arg is the number of miss rate increased times  
                t = threading.Thread(target = env.refresh_clusters, args=(3,))
                t.start()
            elif isinstance(agent, RandomAgent):
                episodes = 2
            else:
                episodes = 1

            for episode in range(episodes):
                # initial observation
                begin = time.time()
                observation = env.reset()
                time1=[]
                time2=[]

                while True:
                    # agent choose action based on observation
                    tick1=time.time()
                    action = agent.choose_action_inference(observation)
                    tick2=time.time()
                    time1.append(tick2-tick1)
                    # agent take action and get next observation and reward
                    observation_, reward = env.step(action, training=False, clustering=False)
                    tick3 = time.time()
                    time2.append(tick3-tick2)
                    # break while loop when end of this episode
                    if env.hasDone():
                        break

                    # swap observation
                    observation = observation_

                    if step % 100 == 0:
                        mr = env.miss_rate()
                        env.notify_miss_rate()
                        #print("Agent=%s, Size=%d, Step=%d, Accesses=%d, Misses=%d, MissRate=%f"
                        #  % (name, cache_size, step, env.total_count, env.miss_count, mr)
                        #)
                    step += 1

                # report after every episode
                end = time.time()
                print("Time1=%f, Time2=%f, Len1=%d"%(np.mean(time1), np.mean(time2), len(time1)))
                mr = env.miss_rate()
                print("Agent=%s, Size=%d, Episode=%d: Accesses=%d, Misses=%d, MissRate=%f, Duration=%f"
                    % (name, cache_size, episode, env.total_count, env.miss_count, mr, end-begin)
                )
                miss_rates.append(mr)
            
            miss_rates = np.array(miss_rates)
            print("Agent=%s, Size=%d: Mean=%f, Median=%f, Max=%f, Min=%f"
                % (name, cache_size, np.mean(miss_rates), np.median(miss_rates), np.max(miss_rates), np.min(miss_rates))
            )
