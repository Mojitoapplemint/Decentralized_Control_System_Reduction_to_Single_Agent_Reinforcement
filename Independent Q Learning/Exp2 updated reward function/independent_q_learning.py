import gymnasium as gym
import numpy as np
import pandas as pd
import cam_env_5050_entry
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
        action = np.random.choice(2)
    else:
        action = np.argmax(q_table[state])
    return action

def get_cat_state_num(observation):
    return CAT_STATES.get(observation)

def get_mouse_state_num(observation):
    return MOUSE_STATES.get(observation)

def independent_q_learning(env, model_name, epochs = 10000, epsilon = 0.1, gamma = 0.9, alpha = 0.9):

    q_cat = np.zeros(shape=(6,2))
    q_mouse = np.zeros(shape=(6,2))

    terminated_count = [0,0,0,0,0]
    truncated_count = [0,0,0,0,0]

    for episode in range(epochs):
        if (episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")

        terminated = False
        truncated = False

        observation, info = env.reset()
        cat_state = get_cat_state_num(observation)
        mouse_state = get_mouse_state_num(observation)

        while not terminated and not truncated:
            mouse_action = get_joint_action(q_mouse, mouse_state, epsilon)
            cat_action = get_joint_action(q_cat, cat_state, epsilon)

            joint_action = (mouse_action, cat_action)
            
            observation, reward, terminated, truncated, info = env.step(joint_action)
            
            #print(reward)
            
            next_mouse_state = get_mouse_state_num(observation)
            next_cat_state = get_cat_state_num(observation)

            q_mouse[mouse_state, mouse_action] = q_mouse[mouse_state, mouse_action] + alpha*(reward + gamma*np.max(q_mouse[next_mouse_state])-q_mouse[mouse_state, mouse_action])
            q_cat[cat_state, cat_action] = q_cat[cat_state, cat_action] + alpha*(reward + gamma*np.max(q_cat[next_cat_state])-q_cat[cat_state, cat_action])

            mouse_state = next_mouse_state
            cat_state = next_cat_state

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
    
    q_mouse_df = pd.DataFrame(q_mouse)
    q_cat_df = pd.DataFrame(q_cat)
    
    q_mouse_df.to_csv(f"C:/Users/woong/Desktop/COMP_SCI/Reinforement Learning/Cat and Mouse/Independent Q Learning/Exp2 updated reward function/{model_name}_mouse.csv")
    q_cat_df.to_csv(f"C:/Users/woong/Desktop/COMP_SCI/Reinforement Learning/Cat and Mouse/Independent Q Learning/Exp2 updated reward function/{model_name}_cat.csv")
    
env = gym.make("CatAndMouse-5050_entry")
independent_q_learning(env, "iql_5050_entry", epochs = 10000, epsilon=0.1, gamma = 0.95)

env = gym.make("CatAndMouse-cat_entry")
independent_q_learning(env, "iql_cat_entry", epochs = 10000, epsilon=0.1, gamma = 0.95)
