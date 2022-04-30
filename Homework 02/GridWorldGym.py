import numpy as np
from enum import Enum 

class GridWorldGym:

    class Actions(Enum):
        TOP = 0
        DOWN = 1
        RIGHT = 2
        LEFT = 3

    def __init__(self):
        self.reset()
        
        
    def reset(self):
        
        self.pos = (0,0)

        self.env = np.zeros(shape=(5,5), dtype=np.uint8)
        self.env[self.pos] = 1
        self.isObstacle = np.zeros(shape=(5,5), dtype=np.bool8)
        self.rewards = np.zeros(shape=(5,5), dtype=np.int8) 
        self.isTerminal = np.zeros(shape=(5,5), dtype=np.bool8)
        self.isTerminal[4,4] = True
        self.rewards[2,2] = -1

        self.isObstacle[3,3] = True
        self.rewards[4,4] = 1
     

        return self.pos 
    
    def getState(self):
        return self.pos 
        
    def step(self, action):
        
        if self.isValidAction(action):
        
            newPos = self.state_transistion_function(self.pos, action)
       
            self.env[self.pos] = 0
            self.env[newPos] = 1

            self.pos = newPos

            return self.pos, self.rewards[self.pos], self.isTerminal[self.pos]
        else:
            raise ValueError("Invalid action")
    

    def state_transistion_function(self, state, action):
        # non-deterministic
        if np.random.rand() < 0.5:
            validActionFound = False 
            while not validActionFound:
                action = np.random.choice(np.arange(self.NUM_ACTIONS))
                validActionFound, _ = self.isValidAction(action)
        
        return self.getNewPos(state, action) # state = position

    def isValidAction(self, action):
            
        newPos = self.getNewPos(self.pos, action)

        if newPos == None:
            return False, None
        
        x,y = newPos
        if x < 0 or y < 0 or x > 4 or y > 4:
            return False, None
        
        if self.isObstacle[x,y]:
            return False, None
        
        return True, newPos

    def getNewPos(self, state, action):
        pos = state 
        if action == self.Actions.TOP.value:
            newPos = (pos[0] - 1, pos[1])    

        elif action == self.Actions.DOWN.value:
            newPos = (pos[0] + 1, pos[1])    

        elif action == self.Actions.RIGHT.value:
            newPos = (pos[0], pos[1] + 1)   

        elif action == self.Actions.LEFT.value:
            newPos = (pos[0], pos[1] - 1)   
        else: 
            newPos = None
        
        return newPos

    def visualize(self):
        for x in range(self.env.shape[0]):
            for y in range(self.env.shape[1]):

                if self.rewards[x,y] == 1:
                    print("$", end=" ")
                
                elif self.rewards[x,y] == -1:
                    print("-", end=" ")
                
                elif self.isObstacle[x,y]:
                    print("☐", end=" ")
                else:
                    print(self.env[x,y], end=" ")
            
            print()
        
        print()
        print("Legend:")
        print("1 = Current position of the agent")
        print("0 = Empty, reward = 0")
        print("$ = Terminal state, reward = 1 ")
        print("- = State to avoid, reward = -1")
        print("☐ = Obstacle")
        print("---------------------------------")

    @property
    def NUM_ACTIONS(self):
        return 4

    @property
    def GRID_SHAPE(self):
        return (5,5)