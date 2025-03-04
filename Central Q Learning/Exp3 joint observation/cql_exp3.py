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

MOUSE_STATES = {
        (1,3):0,
        (1,4):1, (1,5):1,
        (2,3):2,
        (2,4):3, (2,5):3,
        (3,3):4,
        (3,4):5, (3,5):5
    }

CAT_STATES = {
        (1,3):0, (2,3):0,
        (1,4):1, (2,4):1, 
        (1,5):2, (2,5):2, 
        (3,3):3,
        (3,4):4, 
        (3,5):5
    }

def get_joint_action(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(4)
    else:
        action = np.argmax(q_table[state])
    return action

def get_state_number(observation):
    mouse_observation = MOUSE_STATES.get(observation)
    cat_observation = CAT_STATES.get(observation)
    
    return mouse_observation*6 + cat_observation

def central_q_training(env, model_name, epochs = 10000, epsilon = 0.1, gamma = 0.9, alpha = 0.9):

    q_table = np.zeros(shape=(36,4))

    terminated_count = [0,0,0,0,0]
    truncated_count = [0,0,0,0,0]

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

        if episode < epochs/5:
            if terminated:
                terminated_count[0] += 1
            else:
                truncated_count[0] += 1
        elif episode < 2*epochs/5:
            if terminated:
                terminated_count[1] += 1
            else:
                truncated_count[1] += 1
        elif episode < 3*epochs/5:
            if terminated:
                terminated_count[2] += 1
            else:
                truncated_count[2] += 1
        elif episode < 4*epochs/5:
            if terminated:
                terminated_count[3] += 1
            else:
                truncated_count[3] += 1
        else:
            if terminated:
                terminated_count[4] += 1
            else:
                truncated_count[4] += 1
    
    print("Terminated counts")
    for i in range(5):
        print(i*epochs/5,"~",(i+1)*epochs/5 , ":", terminated_count[i])
        
    print("Truncated counts")
    for i in range(5):
        print(i*epochs/5,"~",(i+1)*epochs/5 , ":", truncated_count[i])
    
    df = pd.DataFrame(q_table)
    df.to_csv(f"C:/Users/woong/Desktop/COMP_SCI/Reinforement Learning/Cat and Mouse/Central Q Learning/Exp3 joint observation/{model_name}.csv")



#--- Central Q Learning ---#
env = gym.make("CatAndMouse-cat_entry")

central_q_training(env, "cat_entry_jo", epochs=10000)

env = gym.make("CatAndMouse-5050_entry")

central_q_training(env, "5050_entry_jo", epochs=10000)
