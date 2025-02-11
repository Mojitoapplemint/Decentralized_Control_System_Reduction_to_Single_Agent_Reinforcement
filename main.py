import numpy as np
import pandas as pd
import gymnasium as gym
import cam_env_cat_entry


STATES = {(1,3):0,
          (1,4):1,
          (1,5):2,
          (2,3):3,
          (2,4):4,
          (2,5):5,
          (3,3):6,
          (3,4):7,
          (3,5):8}

def get_state_number(observation, doors):
    state_num = STATES[observation]
    
    door_status_decimal = 2*doors[0]+doors[1]
    
    return 4*state_num+door_status_decimal

q_table = pd.read_csv("test_training.csv")
q_table = q_table.drop(q_table.columns[[0]], axis=1).to_numpy()

env = gym.make("CatAndMouse-v0", render_mode = "human")

terminated = False
truncated = False

observation, info = env.reset()
doors = info["doors"]
env.render()

while(not terminated and not truncated):
    state_num = get_state_number(observation, doors)
    
    action = np.argmax(q_table[state_num])
    
    joint_action = (action//2, action%2)
    
    observation, _, terminated, truncated, info = env.step(joint_action)
    
    doors = info["doors"]


