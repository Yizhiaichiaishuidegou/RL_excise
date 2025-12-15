from scripts.tools import grid_env
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

class Qlearning:
    def __init__(self, env: grid_env.GridEnv):
        self.gamma = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list

        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))

        # off policy
        self.behavior_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.target_policy = np.zeros_like(self.behavior_policy)

        # 初始策略是均匀分配 这里5个动作都是0.2
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象
        # ========== 新增：绘图缓存（用于hold on复用axes） ==========
        self.plot_fig = None  # 缓存绘图的figure
        self.plot_axes = {}   # 缓存子图axes: {subplot_pos: ax}

    def obtain_episode(self,behavior_policy,max_length =200):
        self.env.reset()
        state = self.env.pos2state(self.env.agent_location)
        episode = []
        length = 0
        while length < max_length:
            action = np .random.choice(np.arange(self.action_space_size),p=behavior_policy[state])
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            episode.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })
            # 终止条件：环境done或步数耗尽
            if done:
                break
            state = next_state
            length += 1
        return episode

    def update_behavior_policy(self, epsilon=1.0):
        """行为策略：以mean_policy为核心，轻微偏向最优动作（保留高探索）"""
        for state in range(self.state_space_size):
            # 目标策略：贪心选最优动作
            action_star = self.qvalue[state].argmax()
            # 行为策略：以mean_policy为基础，轻微偏向最优动作
            for a in range(self.action_space_size):
                if a == action_star:
                    self.behavior_policy[state, a] = (1 - epsilon) * 1.0 + epsilon * (1/self.action_space_size)
                else:
                    self.behavior_policy[state, a] = epsilon * (1/self.action_space_size)

    def update_target_policy(self):
        """目标策略：纯贪心（argmax Q）"""
        for state in range(self.state_space_size):
            action_star = self.qvalue[state].argmax()
            self.target_policy[state] = 0.0
            self.target_policy[state, action_star] = 1.0

    def plot_final_fig(self, episode_index_list, reward_list, length_list):
        """训练结束后一次性绘制最终Fig（仅显示一次）"""
        # 创建Fig（仅一次）
        fig = plt.figure(figsize=(10, 10))

        # 子图1：Total Reward（211）
        ax1 = fig.add_subplot(211)
        ax1.plot(episode_index_list, reward_list, color='blue', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Episode Index')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Total Reward per Episode (Off-Policy Q-Learning)')
        ax1.grid(True, alpha=0.3)

        # 子图2：Episode Length（212）
        ax2 = fig.add_subplot(212)
        ax2.plot(episode_index_list, length_list, color='red', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Episode Index')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length per Episode (Off-Policy Q-Learning)')
        ax2.grid(True, alpha=0.3)

        # 调整子图间距，避免重叠
        plt.tight_layout()
        # 显示最终Fig（阻塞，直到关闭窗口）
        plt.show()

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.target_policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self,y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(self.state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)




    def q_learning_on_policy(self, alpha=0.001, epsilon=0.4, num_episodes=1000):
        init_num = num_episodes
        qvalue_list = [self.qvalue, self.qvalue + 1]
        episode_index_list = []
        reward_list = []
        length_list = []


        while num_episodes > 0:
            episode_index_list.append(init_num - num_episodes)
            done = False
            self.env.reset()
            next_state = 0
            total_rewards = 0
            episode_length = 0
            num_episodes -= 1
            print(np.linalg.norm(qvalue_list[-1] - qvalue_list[-2], ord=1), num_episodes)
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                          p=self.policy[state])
                _, reward, done, _, _ = self.env.step(action)
                next_state = self.env.pos2state(self.env.agent_location)
                episode_length += 1
                total_rewards += reward
                next_qvalue_star = self.qvalue[next_state].max()
                target = reward + self.gamma * next_qvalue_star
                error = self.qvalue[state, action] - target
                self.qvalue[state, action] = self.qvalue[state, action] - alpha * error
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, a] = 1 / self.action_space_size * epsilon
            qvalue_list.append(self.qvalue.copy())
            reward_list.append(total_rewards)
            length_list.append(episode_length)
        self.plot_final_fig(episode_index_list, reward_list, length_list)

    def q_learning_off_policy(self, alpha=0.01, num_episodes=2000, max_episode_length=200, epsilon=1.0,
                              explore_decay=0.999):
        """Off-Policy Q-Learning（仅训练结束后显示Fig）"""
        # 初始化数据记录列表（仅存数据，不绘图）
        episode_index_list = []
        reward_list = []
        length_list = []

        current_explore = epsilon  # 初始完全随机（mean_policy）

        for episode_idx in range(num_episodes):
            # 1. 衰减探索性（保留高探索）按照比例衰减 给一个衰减最小阈值
            current_explore = max(0.1, current_explore * explore_decay)
            self.update_behavior_policy(epsilon=current_explore)

            # 2. 采样episode（行为策略生成样本）
            episode = self.obtain_episode(self.behavior_policy, max_length=max_episode_length)

            # 3. 遍历episode更新Q值（核心Off-Policy逻辑）
            total_reward = 0
            for step in episode:
                state = step["state"]
                action = step["action"]
                reward = step["reward"]
                next_state = step["next_state"]
                done = step["done"]
                total_reward += reward

                # Off-Policy Q-Learning核心更新公式
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.qvalue[next_state].max()
                self.qvalue[state, action] += alpha * (target - self.qvalue[state, action])

            # 4. 更新目标策略（贪心）
            self.update_target_policy()

            # 5. 仅记录数据（无任何绘图操作）
            episode_index_list.append(episode_idx)
            reward_list.append(total_reward)
            length_list.append(len(episode))

            # 打印进度（可选）
            if (episode_idx + 1) % 200 == 0:
                avg_reward = np.mean(reward_list[-200:])
                print(
                    f"Episode {episode_idx + 1}/{num_episodes} | 探索度：{current_explore:.3f} | 近200轮平均奖励：{avg_reward:.2f}")

        # ========== 训练结束后：一次性绘制并显示最终Fig ==========
        # self.plot_final_fig(episode_index_list, reward_list, length_list)



if __name__ == "__main__":

    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='human')
    start_time = time.time()

    qlearning = Qlearning(env)
    qlearning.q_learning_off_policy(
        alpha=0.01,
        num_episodes=20000,
        max_episode_length=200,
        epsilon=1.0,  # 初始完全随机（mean_policy）
        explore_decay=0.999
    )

    # qlearning.q_learning_on_policy()

    qlearning.show_policy()
    # qlearning.show_state_value()
    # qlearning.env.render()
    qlearning.env.render_.save_frame('TargetPolicy')

    end_time = time.time()

    print(f"off-policy 训练总耗时：{end_time - start_time:.2f} 秒")