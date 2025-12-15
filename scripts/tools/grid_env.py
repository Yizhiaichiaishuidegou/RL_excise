import time
from typing import Optional, Union, List, Tuple

# 核心替换：gym -> gymnasium
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType

from scripts.tools import render

np.random.seed(1)



def arr_in_list(array, _list):
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


class GridEnv(gym.Env):

    def __init__(self, size: int, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 render_mode: Optional[str] = None):  # 适配gymnasium的render_mode默认值
        """
        GridEnv 的构造函数
        :param size: grid_world 的边长
        :param target: 目标点的pos
        :param forbidden: 不可通行区域 二维数组 或者嵌套列表 如 [[1,2],[2,2]]
        :param render_mode: 渲染模式 video表示保存视频 (None表示不渲染)
        """
        # 初始化可视化
        self.agent_location = np.array([0, 0])
        self.time_steps = 0
        self.size = size
        self.render_mode = render_mode  # gymnasium要求render_mode为Optional[str]
        self.render_ = render.Render(target=target, forbidden=forbidden, size=size)

        # 初始化起点 障碍物 目标点
        self.forbidden_location = []
        for fob in forbidden:
            self.forbidden_location.append(np.array(fob))
        self.target_location = np.array(target)

        # 初始化 动作空间 观测空间（gymnasium与gym兼容，无需修改）
        self.action_space, self.action_space_size = spaces.Discrete(5), spaces.Discrete(5).n
        self.reward_list = [0, 1, -10, -10]
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "barrier": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # action to pos偏移量 的一个map
        self.action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

        # Rsa表示 在 指定 state 选取指点 action 得到reward的概率
        self.Rsa = None
        # Psa表示 在 指定 state 选取指点 action 跳到下一个state的概率
        self.Psa = None
        self.psa_rsa_init()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)  # gymnasium要求显式调用父类reset
        self.agent_location = np.array([0, 0])
        observation = self.get_obs()
        info = self.get_info()
        return observation, info  # gymnasium的reset固定返回 (obs, info)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # 原有逻辑完全保留
        reward = self.reward_list[self.Rsa[self.pos2state(self.agent_location), action].tolist().index(1)]
        direction = self.action_to_direction[action]
        self.render_.upgrade_agent(self.agent_location, direction, self.agent_location + direction)
        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self.agent_location, self.target_location)
        truncated = False  # gymnasium要求返回truncated（截断标志，这里无需截断设为False）
        observation = self.get_obs()
        info = self.get_info()
        return observation, reward, terminated, truncated, info  # 适配gymnasium的step返回值

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))
        if self.render_mode is not None:  # 避免空渲染模式调用show_frame
            self.render_.show_frame(100000)
        return None

    def get_obs(self) -> ObsType:
        return {"agent": self.agent_location, "target": self.target_location, "barrier": self.forbidden_location}

    def get_info(self) -> dict:
        return {"time_steps": self.time_steps}

    def state2pos(self, state: int) -> np.ndarray:
        return np.array((state // self.size, state % self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        """
        初始化网格世界的 psa 和 rsa
        保留原有注释和逻辑
        """
        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(self.size ** 2, self.action_space_size, len(self.reward_list)), dtype=float)
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                pos = self.state2pos(state_index)
                next_pos = pos + self.action_to_direction[action_index]

                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:
                    self.Psa[state_index, action_index, state_index] = 1
                    self.Rsa[state_index, action_index, 3] = 1

                else:
                    self.Psa[state_index, action_index, self.pos2state(next_pos)] = 1
                    if np.array_equal(next_pos, self.target_location):
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location):
                        self.Rsa[state_index, action_index, 2] = 1
                    else:
                        self.Rsa[state_index, action_index, 0] = 1

    def close(self):
        """gymnasium要求实现close方法，释放资源"""
        pass


if __name__ == "__main__":
    # 测试代码适配：render_mode='' 改为 render_mode=None（符合gymnasium规范）
    grid = GridEnv(size=5, target=[1, 2], forbidden=[[2, 2]], render_mode=None)
    obs, info = grid.reset()  # 适配reset返回值
    grid.render()

    # 可选：测试step接口
    # action = 1  # 向右移动
    # obs, reward, terminated, truncated, info = grid.step(action)
    # print(f"观测：{obs}, 奖励：{reward}, 是否到达目标：{terminated}")
