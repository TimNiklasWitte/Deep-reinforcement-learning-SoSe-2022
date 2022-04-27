import numpy as np
from enum import Enum 

class GridWorldGym:

    class Actions(Enum):
        TOP = 1
        DOWN = 2
        RIGHT = 3
        LEFT = 4


    def __init__(self):
        self.reset()
        
        
    def reset(self):
        
        self.pos = (0,0)

        self.env = np.zeros(shape=(5,5), dtype=np.uint8)
        self.env[self.pos] = 1
        self.isObstacle = np.zeros(shape=(5,5), dtype=np.bool8)
        self.rewards = np.zeros(shape=(5,5), dtype=np.uint8) 
        self.isTerminal = np.zeros(shape=(5,5), dtype=np.bool8)

        self.isObstacle[3,3] = True
        self.rewards[4,4] = 1
        self.rewards[4,4] = True

    def step(self, action):
        
        if self.isValidAction(action):
            newPos = self.getNewPos(action)
       
            self.env[self.pos] = 0
            self.env[newPos] = 1

            self.pos = newPos

            return self.pos, self.rewards[self.pos], self.isTerminal[self.pos]
        else:
            raise ValueError("Invalid action")

    def isValidAction(self, action):
        
        try:
            newPos = self.getNewPos(action)
        except ValueError:
            return False
        
        x,y = newPos
        if x < 0 or y < 0 or x > 4 or y > 4:
            return False
        
        if self.isObstacle[x,y]:
            return False
        
        return True, newPos

    def getNewPos(self, action):
       
        if action == self.Actions.TOP:
            newPos = (self.pos[0] - 1, self.pos[1])    

        elif action == self.Actions.DOWN:
            newPos = (self.pos[0] + 1, self.pos[1])    

        elif action == self.Actions.RIGHT:
            newPos = (self.pos[0], self.pos[1] + 1)   

        elif action == self.Actions.LEFT:
            newPos = (self.pos[0], self.pos[1] - 1)   
        else: 
            raise ValueError("Invalid action")
        
        return newPos

    def visualize():
        pass