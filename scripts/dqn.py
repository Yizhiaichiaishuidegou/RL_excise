import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

from scripts.tools import grid_env
from scripts.tools.model import *



class DQN:
    def __init__(self,env:grid_env.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        # 初始策略是均匀分配 这里5个动作都是0.2你
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")

    def obtain_episode(self, policy, start_state, start_action, length):
        f"""

        :param policy: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :param length: episode 长度
        :return: 一个 state,action,reward,next_state,next_action 序列
        """
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
    def get_data_iter(self, episode, batch_size=64, is_train=True):
        """构造一个PyTorch数据迭代器"""
        reward = []
        state_action = []
        next_state = []
        for i in range(len(episode)):
            reward.append(episode[i]['reward'])
            action = episode[i]['action']
            y, x = self.env.state2pos(episode[i]['state'])
            state_action.append((y, x, action))
            y, x = self.env.state2pos(episode[i]['next_state'])
            next_state.append((y, x))
        reward = torch.tensor(reward).reshape(-1, 1)
        state_action = torch.tensor(state_action)
        next_state = torch.tensor(next_state)
        data_arrays = (state_action, reward, next_state)
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)

    def dqn(self, learning_rate=0.0015, episode_length=5000, epochs=600, batch_size=100, update_step=10):

        q_net = QNET()
        policy = self.policy.copy()
        state_value = self.state_value.copy()

        q_target_net = QNET()
        q_target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.SGD(q_net.parameters(),
                                    lr=learning_rate)
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=episode_length)
        date_iter = self.get_data_iter(episode, batch_size)
        loss = torch.nn.MSELoss()
        approximation_q_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        i = 0
        rmse_list =[]
        loss_list =[]
        for epoch in range(epochs):
            for state_action, reward, next_state in date_iter:
                i += 1
                q_value = q_net(state_action)
                q_value_target = torch.empty((batch_size, 0))  # 定义空的张量
                for action in range(self.action_space_size):
                    s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
                    q_value_target = torch.cat((q_value_target, q_target_net(s_a)), dim=1)
                q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]
                y_target_value = reward + self.gama * q_star
                l = loss(q_value, y_target_value)
                optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0S
                l.backward()  # 反向传播更新参数
                optimizer.step()
                if i % update_step == 0 and i != 0:
                    q_target_net.load_state_dict(
                        q_net.state_dict())  # 更新目标网络
                    # policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            loss_list.append(float(l))
            print("loss:{},epoch:{}".format(l, epoch))
            self.policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            self.state_value = np.zeros(shape=self.state_space_size)

            for s in range(self.state_space_size):
                y, x = self.env.state2pos(s)
                for a in range(self.action_space_size):
                    approximation_q_value[s, a] = float(q_net(torch.tensor((y, x, a)).reshape(-1, 3)))
                q_star_index = approximation_q_value[s].argmax()
                self.policy[s, q_star_index] = 1
                self.state_value[s] = approximation_q_value[s, q_star_index]
            rmse_list.append(np.sqrt(np.mean((state_value - self.state_value) ** 2)))
            # policy_rmse = np.sqrt(np.mean((policy - self.policy) ** 2))
        fig_rmse = plt.figure(figsize=(8, 12))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(211)

        # 绘制 rmse 图像
        ax_rmse.plot(rmse_list)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        self.writer.close()
        ax_loss = fig_rmse.add_subplot(212)

        ax_loss.plot(loss_list)
        ax_loss.set_title('loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        plt.show()

if __name__ =="__main__":

    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    Dqn = DQN(env)
    Dqn.dqn()
