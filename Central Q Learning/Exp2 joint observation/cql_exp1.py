import sys
sys.path.insert(0, "C:/Users/woong/Desktop/COMP_SCI/Reinforement Learning/Cat and Mouse/Central Q Learning")


import gymnasium as gym
import numpy as np
import pandas as pd
import cam_env_cat_entry
import cam_env_5050_entry

STATES = {(1,3):0,
          (1,4):1,
          (1,5):2,
          (2,3):3,
          (2,4):4,
          (2,5):5,
          (3,3):6,
          (3,4):7,
          (3,5):8}

def get_joint_action(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(4)
    else:
        action = np.argmax(q_table[state])
    return action

def get_state_number(observation):
    state_num = STATES[observation]
    
    return state_num

def central_q_learning(env, model_name, epochs = 10000, epsilon = 0.1, gamma = 0.9, alpha = 0.9):

    q_table = np.zeros(shape=(9,4))

    for episode in range(epochs):
        if (episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")

        terminated = False
        truncated = False
        
        observation, info = env.reset()
        
        new_state = get_state_number(observation)
        
        while (not terminated and not truncated):
        
            action = get_joint_action(q_table, new_state , epsilon)
            
            joint_action = (action//2, action%2)
            
            old_state = new_state
            
            observation, reward, terminated, truncated, info = env.step(joint_action)
            
            new_state = get_state_number(observation)
            
            q_table[old_state, action] = q_table[old_state, action] + alpha*(reward + gamma*np.max(q_table[new_state]) - q_table[old_state, action])
    
    df = pd.DataFrame(q_table)
    df.to_csv(f"C:/Users/woong/Desktop/COMP_SCI/Reinforement Learning/Cat and Mouse/Central Q Learning/Exp2 joint observation/{model_name}.csv")



#--- Central Q Learning ---#
env = gym.make("CatAndMouse-cat_entry")

central_q_learning(env, "cat_entry", epochs=10000)

env = gym.make("CatAndMouse-5050_entry")

central_q_learning(env, "5050_entry", epochs=10000)
