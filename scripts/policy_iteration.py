import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

from scripts.tools import grid_env


class Policy_iteration:
    def __init__(self, env: grid_env.GridEnv):

        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        # 初始策略是均匀分配 这里5个动作都是0.2
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

    def random_greed_policy(self):
        """
        生成随机的greedy策略
        :return:
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state_index, action] = 1
        return policy

    def calculate_qvalue(self, state, action, state_value):
        """
        计算qvalue elementwise形式
        :param state: 对应的state
        :param action: 对应的action
        :param state_value: 状态值
        :return: 计算出的结果
        """
        qvalue = 0
        for i in range(self.reward_space_size):
            qvalue += self.reward_list[i] * self.env.Rsa[state, action, i] # sum (r*p(r|s,a)) for every r ,which this action can grasp immediately reward for r set
        for next_state in range(self.state_space_size):
            qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value[next_state]# for deterministic policy q_value = state value
        return qvalue
    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
        """
        迭代求解贝尔曼公式 得到 state value tolerance 和 steps 满足其一即可
        :param policy: 需要求解的policy
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止计算 此时若是policy iteration 则算法变为 truncated iteration
        :return: 求解之后的收敛值
        """
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > tolerance:
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):# state value = sum π(a|s)q_π（s,a） for every (s,a) pair
                    value += policy[state, action] * self.calculate_qvalue(state_value=state_value_k.copy(),
                                                                           state=state,
                                                                           action=action)  # bootstrapping
                state_value_k[state] = value
        return state_value_k
    #经过policy evaluation 后得到π0 下的state value 现在计算每个state下面的 action value

    def policy_improvement(self, state_value):
        """
        是普通 policy_improvement 的变种 相当于是值迭代算法 也可以 供策略迭代使用 做策略迭代时不需要 接收第二个返回值
        更新 qvalue ；qvalue[state,action]=reward+value[next_state]
        找到 state 处的 action*：action* = arg max(qvalue[state,action]) 即最优action即最大qvalue对应的action
        更新 policy ：将 action*的概率设为1 其他action的概率设为0 这是一个greedy policy
        :param: state_value: policy对应的state value
        :return: improved policy, 以及迭代下一步的state_value
        """
        # 初始化新的policy
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        # 拷贝 Policy evaluation 生成的 state value
        state_value_k = state_value.copy()

        for state in range(self.state_space_size):
            # 对每个状态计算采取不同 action 的不同的 k+1 步 state value
            qvalue_list = []
            for action in range(self.action_space_size):
                qvalue_list.append(self.calculate_qvalue(state, action, state_value.copy()))
            # 选择最大的当这一步improvement 后的state value 最大的action 为 k+1策略的action
            state_value_k[state] = max(qvalue_list)
            action_star = qvalue_list.index(max(qvalue_list))
            # 更新策略
            policy[state, action_star] = 1
        return policy, state_value_k

    def policy_iteration(self, tolerance=0.001, steps=100):
        """

        :param tolerance: 迭代前后policy的范数小于tolerance 则认为已经收敛
        :param steps: step 小的时候就退化成了  truncated iteration
        :return: 剩余迭代次数
        """
        policy = self.random_greed_policy()
        # 策略最优 判定
        while np.linalg.norm(policy - self.policy, ord=1) > tolerance and steps > 0:
            steps -= 1
            policy = self.policy.copy()
            # 这里给的事初始化策略
            # PE 策略评估计算k 个策略的state value
            self.state_value = self.policy_evaluation(self.policy.copy(), tolerance, steps)
            # PI 策略优化计算每个 state  的 action value ,选择最大的作为这个state 的action =====> new policy
            self.policy, _ = self.policy_improvement(self.state_value)
        return steps

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

    start_time = time.time()
    policyiteration = Policy_iteration(env)
    policyiteration.policy_iteration(tolerance=0.001,steps = 500)
    policyiteration.show_policy()
    policyiteration.env.render_.save_frame('PolicyIteration')
    start_time = time.time()