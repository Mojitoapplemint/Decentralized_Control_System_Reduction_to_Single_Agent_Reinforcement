import gymnasium as gym
import numpy as np
from gymnasium import spaces
from IPython.display import clear_output
import time

gym.register(
    id="CatAndMouse-v0",
    entry_point="cat_and_mouse_env:CatAndMouseEnv"
)

class CatAndMouseEnv(gym.Env):
    """
    Initialize the CatAndMouseEnv environment.
    Args:
        render_mode (str, optional): The mode to render the environment. Defaults to None.
    """
    metadata = {"render_modes": ["human"], "render_fps":4}
    
    
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low= 1, high=5, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
    
    def cat_position_to_door(self):
        if self.cat_position == 3:
            cat_door = "c3"
        elif self.cat_position == 4:
            cat_door = "c1"
        elif self.cat_position == 5:
            cat_door = "c2"
        return cat_door

    def mouse_position_to_door(self):
        if self.mouse_position == 3:
            mouse_door = "m3"
        elif self.mouse_position == 2:
            mouse_door = "m1"
        elif self.mouse_position == 1:
            mouse_door = "m2"
        return mouse_door
    
    def update_door(self, mouse_action, cat_action):
        cat_door = self.cat_position_to_door()
        mouse_door = self.mouse_position_to_door()
        
        self.doors[cat_door] = cat_action
        self.doors[mouse_door] = mouse_action
        
    def mouse_move(self):
        mouse_door = self.mouse_position_to_door()
        
        if self.doors[mouse_door] == 1:
            if self.mouse_position == 1:
                self.mouse_position = 3
                self.mouse_room3 = True
            else:
                self.mouse_position -=1
        
    def cat_move(self):
        cat_door = self.cat_position_to_door()
        
        if self.doors[cat_door] == 1:
            if self.cat_position == 5:
                self.cat_position = 3
                self.cat_room3 = True
            else:
                self.cat_position +=1
    
    def move(self):
        mouse_door = self.mouse_position_to_door()
        cat_door = self.cat_position_to_door()
        
        if self.doors[mouse_door] == 1:
            if self.mouse_position == 1:
                self.mouse_position = 3
                self.mouse_room3 = True
            else:
                self.mouse_position -=1
        
        if self.doors[cat_door] == 1:
            if self.cat_position == 5:
                self.cat_position = 3
                self.cat_room3 = True
            else:
                self.cat_position +=1    
        
    def reset(self, seed=None, options = None):
        """
        Reset the environment to its initial state.
        Returns:
            list: [cat position, mouse position].
            dict: A dictionary containing the current status for each door, in a single list.
        """
        self.cat_position = 4
        self.mouse_position = 2
        self.cat_room3 = False;
        self.mouse_room3 = False;
        self.doors = {"m1":1,
                      "m2":1,
                      "m3":1,
                      "c1":1,
                      "c2":1,
                      "c3":1}
        
        self.training_count = 0;
        
        info = {"doors": (1, 1)}
        observation = (self.mouse_position, self.cat_position)
        return observation, info
    
    
    def step(self, joint_action):
        reward = 0
        terminated = False
        truncated = False
        
        mouse_action , cat_action= joint_action
        self.update_door(mouse_action, cat_action)
        
        self.cat_move()
        if self.render_mode == "human":
            self.render2("Cat")
        if self.cat_position == 3 and self.mouse_position==3:
            reward = -100
            terminated = True
        else:
            self.mouse_move()
            if self.render_mode == "human":
                self.render2("Mouse")
            
            if self.cat_position == 3 and self.mouse_position==3:
                reward = -100
                terminated = True
            elif self.cat_room3 and self.mouse_room3:
                reward = 10
                self.cat_room3 = False
                self.mouse_room3 = False
            
            if self.training_count == 30:
                truncated = True
            else:
                self.training_count += 1
        
        
        cat_door = self.doors[self.cat_position_to_door()]
        mouse_door = self.doors[self.mouse_position_to_door()]
        info = {"doors": (mouse_door, cat_door)}
        observation = (self.mouse_position, self.cat_position)
        
        return observation, reward, terminated, truncated, info
        

    def render(self):
        """
        Render the environment.
        """
        cat = self.cat_position-1
        mouse = self.mouse_position-1
        if self.cat_position == 3:
            cat=5

        grid = [" " for _ in range(6)]
        grid[cat] = "C"
        grid[mouse] = "M"
        time.sleep(1)
        clear_output(wait=True)
        print(f"\
    _________________________\n\
    |   1   |       |   4   |\n\
    |   {grid[0]}   |   3   |   {grid[3]}   |\n\
    ---------   {grid[2]}   ---------\n\
    |   2   |   {grid[5]}   |   5   |\n\
    |   {grid[1]}   |       |   {grid[4]}   |")
        
    
    def render2(self, cat_or_mouse):
        """
        Render the environment.
        """
        cat = self.cat_position-1
        mouse = self.mouse_position-1
        if self.cat_position == 3:
            cat=5

        grid = [" " for _ in range(6)]
        grid[cat] = "C"
        grid[mouse] = "M"
        
        door_list = ["m1","m2","m3","c1","c2","c3"]
        
        doors = [" " for _ in range(6)]
        
        for i in range(6):
            if self.doors[door_list[i]] == 0:
                doors[i] = "X"
        
        time.sleep(1)
        clear_output(wait=True)
        
        print(f"{cat_or_mouse}'s Move")
        print(f"\
    _________________________\n\
    |   1   |       |   4   |\n\
    |   {grid[0]}   {doors[1]}   3   {doors[5]}   {grid[3]}   |\n\
    |       |       |       |               \n\
    ----{doors[0]}----   {grid[2]}   ----{doors[3]}----\n\
    |   2   |   {grid[5]}   |   5   |\n\
    |   {grid[1]}   {doors[2]}       {doors[4]}   {grid[4]}   |\n\
    |       |       |       |               ")
        print(doors)
        
# self.mouse_move()
#         if self.render_mode == "human":
#             self.render2()
        
#         if self.cat_position == 3 and self.mouse_position==3:
#             reward = -100
#             terminated = True
#         else:
#             self.cat_move()
#             if self.render_mode == "human":
#                 self.render2()
            
#             if self.cat_position == 3 and self.mouse_position==3:
#                 reward = -100
#                 terminated = True
#             elif self.cat_room3 and self.mouse_room3:
#                 reward = 10
#                 self.cat_room3 = False
#                 self.mouse_room3 = False
            
#             if self.training_count == 30:
#                 truncated = True
#             else:
#                 self.training_count += 1