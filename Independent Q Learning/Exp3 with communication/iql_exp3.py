import gymnasium as gym
import numpy as np
import pandas as pd
import cam_env_5050_entry
import cam_env_cat_entry

MOUSE_STATES = {
        (1,3):0,
        (1,4):4, (1,5):4,
        (2,3):8,
        (2,4):12, (2,5):12,
        (3,3):16,
        (3,4):20, (3,5):20
    }

MOUSE_DOORS = ["m1", "m2", "m3", "c2", "c3"]

CAT_STATES = {
        (1,3):0, (2,3):0,
        (1,4):4, (2,4):4, 
        (1,5):8, (2,5):8, 
        (3,3):12,
        (3,4):16, 
        (3,5):20
    }

CAT_DOORS = ["m2", "m3", "c1", "c2", "c3"]

def get_joint_action(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(2)
    else:
        action = np.argmax(q_table[state])
    return action

def get_cat_state_num(observation, door_projection):
    return CAT_STATES.get(observation)+binary_tuple_to_decimal(door_projection)


def get_mouse_state_num(observation, door_projection):
    return MOUSE_STATES.get(observation)+binary_tuple_to_decimal(door_projection)


def binary_tuple_to_decimal(door_status):
    return door_status[0]*2 + door_status[1]


def independent_q_learning(env, model_name, epochs = 10000, epsilon = 0.1, gamma = 0.9, alpha = 0.9):

    q_cat = np.zeros(shape=(24,2))
    q_mouse = np.zeros(shape=(24,2))

    terminated_count = [0,0,0,0,0]
    truncated_count = [0,0,0,0,0]

    for episode in range(epochs):
        if (episode%100==0):
            print(str(100*episode/epochs)+"%","done" , end="\r")

        terminated = False
        truncated = False

        observation, info = env.reset()
        
        doors = info.get("doors")
        
        mouse_state = get_mouse_state_num(observation, (1,1))
        cat_state = get_cat_state_num(observation, (1,1))

        while not terminated and not truncated:
            mouse_action = get_joint_action(q_mouse, mouse_state, epsilon)
            cat_action = get_joint_action(q_cat, cat_state, epsilon)

            joint_action = (mouse_action, cat_action)
            
            observation, reward, terminated, truncated, info = env.step(joint_action)
        
            doors = info.get("doors")
            next_mouse_state = get_mouse_state_num(observation, joint_action)
            next_cat_state = get_cat_state_num(observation, joint_action)

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
    
    q_mouse_df.to_csv(f"C:/Users/woong/Desktop/COMP_SCI/Reinforement Learning/Cat and Mouse/Independent Q Learning/Exp3 with communication/{model_name}_mouse.csv")
    q_cat_df.to_csv(f"C:/Users/woong/Desktop/COMP_SCI/Reinforement Learning/Cat and Mouse/Independent Q Learning/Exp3 with communication/{model_name}_cat.csv")
    
env = gym.make("CatAndMouse-5050_entry")
independent_q_learning(env, "iql_5050_entry",epochs=100000, epsilon=0.1, gamma = 0.8, alpha = 0.9)

env = gym.make("CatAndMouse-cat_entry")
independent_q_learning(env, "iql_cat_entry", epochs = 100000, epsilon=0.1, gamma = 0.8, alpha = 0.9)
