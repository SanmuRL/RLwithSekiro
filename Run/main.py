import torch.nn as nn
import torch
import cv2
import matplotlib.pyplot as plt
from Agents.Q_learning_agents.DQN import DQN
from Agents.Q_learning_agents.DDQN import DDQN
from Utilities.Get_Screen.get_screen import grab_screen, get_blood
from Environments.env import step, restart
from Run.action_set import attack
from Utilities.key_set.directkeys import key_check
from collections import deque
import numpy as np
import time

config = {"gamma": 0.99, "step": 0.001, "capacity": 2000,
          "batch_size": 16, "img_stack": 4, "action_dim": 6,
          "save_path_local": "D:/PythonPro/Myproject/FirstTry/ModelPara/local.pth",
          "save_path_target": "D:/PythonPro/Myproject/FirstTry/ModelPara/target.pth",
          "useExit": 0, "turn_off_exploration": 0, "exploration_cycle_length": None,
          "epsilon_decay_denominator": 15}

if __name__ == "__main__":
    paused = True
    while True:
        if paused:
            keys = key_check()
            if 'T' in keys:
                paused = False
                print("start")
                break
    agent = DDQN(config=config)
    fig = plt.figure()
    fig.suptitle('DDQN', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("episodes")
    ax.set_ylabel("rewards")
    Eposide = 50
    total_reward = 0
    episode_steps = 0
    episode_reward = 0
    episode_num = 0
    episode_rewards = []
    episode_nums = []
    for episode in range(Eposide):
        t = time.time()
        while True:
            img = grab_screen((8, 0, 1450, 849))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self_blood_last = get_blood(img[765:805, 82:561], 10, 90, 100) / 4.55
            enemy_blood_last = get_blood(img[106:146, 83:405], 5, 65, 77) / 2.51
            if self_blood_last > 2:
                break
            if time.time() - t > 2:
                attack()
                time.sleep(0.5)
        state_single = img[94:794, 446:980]
        state = deque(maxlen=agent.img_stack)
        for _ in range(agent.img_stack):
            state.append(state_single)
        self_blood_last = get_blood(img[765:805, 82:561], 10, 90, 100) / 4.55
        enemy_blood_last = get_blood(img[106:146, 83:405], 5, 65, 77) / 2.51
        while True:
            episode_steps += 1
            action = agent.choose_action(state, episode_num)
            reward, next_state, done, self_blood_last, enemy_blood_last = step(action, self_blood_last, enemy_blood_last, state)
            episode_reward += reward
            agent.replay_buffer.add_experience(state, action, reward, next_state, done)
            if agent.replay_buffer.__len__() > agent.batch_size:
                agent.update_local_network()
                agent.soft_target_update(0.01)
            state = next_state
            if done:
                break
        episode_num += 1
        episode_nums.append(episode_num)
        episode_rewards.append(episode_reward)
        print("current episode_num: ", episode_num, "episode_reward", episode_reward, "episode_steps", episode_steps)
        if episode_num % 30 == 0:
            agent.locally_save_policy()
        episode_reward = 0
        episode_steps = 0
        time.sleep(6)
        restart()
    rewards = np.array(episode_rewards)
    nums = np.array(episode_nums)
    np.savetxt("D:/PythonPro/Myproject/FirstTry/ModelPara/rewards.txt", (rewards, nums), fmt='%0.8f')
    ax.plot(episode_steps, episode_rewards)
    ax.show()
