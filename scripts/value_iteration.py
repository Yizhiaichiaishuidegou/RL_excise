import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

from scripts.tools import grid_env



class Value_iteration:
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
            #选择最大的当这一步improvement 后的state value 最大的action 为 k+1策略的action
            state_value_k[state] = max(qvalue_list)
            action_star = qvalue_list.index(max(qvalue_list))
            # 更新策略
            policy[state, action_star] = 1
        return policy, state_value_k

    def value_iteration(self, tolerance=0.001, steps=100):
        """
        迭代求解最优贝尔曼公式 得到 最优state value tolerance 和 steps 满足其一即可
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止 建议将此变量设置大一些
        :return: 剩余迭代次数
        """
        state_value_k = np.ones(self.state_space_size)
        # state value 不变判定 到达不动点
        while np.linalg.norm(state_value_k - self.state_value, ord=1) > tolerance and steps > 0:
            steps -= 1
            self.state_value = state_value_k.copy()
            # 这里只进行一步迭代 这里给的是初始化 state value
            self.policy, state_value_k = self.policy_improvement(state_value_k.copy())
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
    value_iteration = Value_iteration(env)
    value_iteration.value_iteration(tolerance=0.001, steps=500)
    value_iteration.show_policy()
    value_iteration.show_state_value(value_iteration.state_value, y_offset=0.2)
    value_iteration.env.render_.save_frame('value_iteration')
    end_time = time.time()
    print("Total time: ", end_time - start_time)