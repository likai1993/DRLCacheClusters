#! /usr/bin/python3
import sys,time
from cache.Cache_Random import Cache_Random
from cache.Cache_LRU import Cache_LRU
from cache.Cache_LFU import Cache_LFU
from agents.ReflexAgent import *
from cache.DataLoader import * 
from cache.Cache import Cache

if __name__ == "__main__":
    # disk activities
    datafile = sys.argv[1]
    if len(sys.argv) == 4:
        datafile2 = sys.argv[2]
        num_of_peroids = int(sys.argv[3])
        dataloader = DataLoaderMix([datafile, datafile2], num_of_peroids) 
    else:
        dataloader = DataLoaderPintos([datafile])
    
    #sizes = [5, 25, 50, 100, 300]
    sizes = [10000, 50000, 100000, 200000]
    sizes = [7500]
    for cache_size in sizes:
        
        print("==================== Cache Size: %d ====================" % cache_size)
        # cache
        env = Cache(dataloader, cache_size
            , feature_selection=('Base',)
            , reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
            , allow_skip=False)
        
        # agents
        agents = {}
        agents['Random'] = RandomAgent(env.n_actions)
        agents['LRU'] = LRUAgent(env.n_actions)
        agents['LFU'] = LFUAgent(env.n_actions)
        agents['MRU'] = MRUAgent(env.n_actions)
    
        for (name, agent) in agents.items():

            if isinstance(agent, LRUAgent) or isinstance(agent, MRUAgent):
                env = Cache_LRU(dataloader, cache_size
                , feature_selection=('Base',)
                , reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
                , allow_skip=False)
            elif isinstance(agent, LFUAgent):
                env = Cache_LFU(dataloader, cache_size
                , feature_selection=('Base',)
                , reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
                , allow_skip=False)
            elif isinstance(agent, RandomAgent):
                env = Cache_Random(dataloader, cache_size
                , feature_selection=('Base',)
                , reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
                , allow_skip=False)
        
            print("-------------------- %s --------------------" % name)
            step = 0
            miss_rates = []    # record miss rate for every episode
            
            # determine how many episodes to proceed
            # 100 for learning agents, 20 for random agents
            # 1 for other agents because their miss rates are invariant
            if isinstance(agent, RandomAgent):
                episodes = 2
            else:
                episodes = 1

            for episode in range(episodes):
                # initial observation
                observation = env.reset()
                time1=[]
                time2=[]
                time3=[]
                time4=[]
                begin=time.time()
                while True:
                    # agent choose action based on observation
                    tick1 = time.time()
                    action = agent.choose_action(observation)
                    tick2 = time.time()
                    time1.append(tick2-tick1)

                    # agent take action and get next observation and reward
                    observation_, reward = env.step(action)
                    tick3 = time.time()
                    time2.append(tick3-tick2)

                    # break while loop when end of this episode
                    if env.hasDone():
                        break

                    # swap observation
                    observation = observation_

                    if step % 100 == 0:
                        mr = env.miss_rate()
                        #print("Agent=%s, Size=%d, Step=%d, Accesses=%d, Misses=%d, MissRate=%f"
                        #  % (name, cache_size, step, env.total_count, env.miss_count, mr)
                        #)
                    step += 1

                # report after every episode
                end=time.time()
                print("Time1=%f, Time2=%f, Len1=%d" %(np.mean(time1), np.mean(time2), len(time1)))
                mr = env.miss_rate()
                print("Agent=%s, Size=%d, Episode=%d: Accesses=%d, Misses=%d, MissRate=%f, Duration=%f"
                    % (name, cache_size, episode, env.total_count, env.miss_count, mr, end-begin)
                )
                miss_rates.append(mr)

            # summary
            miss_rates = np.array(miss_rates)
            print("Agent=%s, Size=%d: Mean=%f, Median=%f, Max=%f, Min=%f"
                % (name, cache_size, np.mean(miss_rates), np.median(miss_rates), np.max(miss_rates), np.min(miss_rates))
            )
