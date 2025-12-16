
from scripts.tools import grid_env
import time
import random


import numpy as np


class MonteCarlo:
    def __init__(self, env:grid_env.GridEnv):
        self.gamma = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2
        self.reward_space_szie ,self.reward_list =  len(self.env.reward_list),self.env.reward_list

        self.state_value = np.zeros(shape = self.state_space_size)
        self.action_value = np.zeros(shape = (self.state_space_size,self.action_space_size))
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape = (self.state_space_size,self.action_space_size))/self.action_space_size
        self.policy = self.mean_policy.copy()


    def obtain_episode(self,policy,start_state,start_action,length):
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode

    def MC_basic(self,episode_num = 100,length= 25,epochs = 100):

        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    g = 0
                    for episode_ in range(episode_num):
                        episode = self.obtain_episode(self.policy, state, action, length)
                        g_ = 0
                        for step in range(len(episode) - 1, -1, -1):
                            g_ = episode[step]['reward'] + self.gamma * g_
                        g += g_
                    g = g/episode_num
                    self.qvalue[state][action] = g
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                self.policy[state] = np.zeros(shape=self.action_space_size)
                self.policy[state, action_star] = 1



    def MC_exploring_start(self,length = 300,every_visit = 0):

        initial_policy = np.zeros(shape = (self.state_space_size, self.action_space_size))
        initial_policy[:,0]= 1
        behavior_policy = initial_policy
        qvalue = self.qvalue.copy()
        returns = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        while np.linalg.norm(behavior_policy - self.policy,ord =1) > 1e-3:
            behavior_policy = self.policy.copy()
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    visit_list = []
                    g = 0
                    visit_nums = 0
                    episcode = self.obtain_episode(policy = behavior_policy,start_state = state,
                                                   start_action=action,length =length)
                    for step in range(len(episcode) - 1, -1, -1):
                        reward = episcode[step]['reward']
                        state = episcode[step]['state']
                        action = episcode[step]['action']
                        g =self.gamma*g + reward
                        # first visit
                        if every_visit == 0:
                            if[state,action] not in visit_list:
                                visit_list.append([state,action])
                                returns[state][action].append(g)
                                qvalue[state][action] = np.array(returns[state][action]).mean()
                                qvalue_star = qvalue[state].max()
                                action_star = qvalue[state].tolist().index(qvalue_star)
                                self.policy[state] = np.zeros(shape=self.action_space_size)
                                self.policy[state, action_star] = 1
                        else:
                            returns[state][action].append(g)
                            qvalue[state][action] = np.array(returns[state][action])
                            qvalue_star = qvalue[state].max()
                            action_star = qvalue[state].tolist().index(qvalue_star)
                            self.policy[state][action] = np.zeros(shape=self.action_space_size)
                            self.policy[state, action_star] = 1

            print(np.linalg.norm(behavior_policy - self.policy,ord =1))

    def MC_epsilon_greeedy(self,length = 1000,epsilon = 0.5,tolerance = 0.1):

        norm_list =[]
        returns = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        while 1:
            if epsilon <=0 or epsilon >= 1:
                print("epsilon excel bounding!")
                break
            if len(norm_list) >= 3:
                if norm_list[-1] < tolerance and norm_list[-2] < tolerance and norm_list[-3] < tolerance:
                    print("target policy converged!")
                    break

            qvalue = self.qvalue.copy()

            state = random.choice(range(self.state_space_size))

            action = random.choice(range(self.action_space_size))

            episode = self.obtain_episode(policy=self.policy, start_state=state, start_action=action,
                                          length=length)
            g= 0
            for step in range(len(episode) - 1, -1, -1):
                reward = episode[step]['reward']
                state = episode[step]['state']
                action = episode[step]['action']
                g =self.gamma*g + reward
                returns[state][action].append(g)
                self.qvalue[state][action] = np.array(returns[state][action]).mean()

                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state,a] = 1- ((self.action_space_size-1)/self.action_space_size)*epsilon
                    else:
                        self.policy[state,a] = epsilon/self.action_space_size

            print(np.linalg.norm(self.qvalue - qvalue, ord=1))
            norm_list.append(np.linalg.norm(self.qvalue - qvalue, ord=1))

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)






if __name__ == "__main__":
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='human')
    start = time.time()
    montecarlo = MonteCarlo(env=env)

    #montecarlo.MC_basic(episode_num = 100,length= 25,epochs = 100)
    #montecarlo.MC_exploring_start(length = 30,every_visit = 0)
    montecarlo.MC_epsilon_greeedy(length = 1000,epsilon = 0.01,tolerance = 0.2)

    montecarlo.show_state_value(montecarlo.state_value, y_offset=0.2)
    montecarlo.show_policy()
    montecarlo.env.render_.save_frame('MonteCarlo_epsilon_greedy')
    end = time.time()

    print(" Montecarlo Total time: ", end - start)

