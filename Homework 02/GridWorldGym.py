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
        
        self.state = (0,0)

        self.env = np.zeros(shape=(5,5), dtype=np.uint8)
        self.env[self.state] = 1
        self.isObstacle = np.zeros(shape=(5,5), dtype=np.bool8)
        self.rewards = np.zeros(shape=(5,5), dtype=np.int8) 
        self.isTerminal = np.zeros(shape=(5,5), dtype=np.bool8)
        self.isTerminal[4,4] = True
        self.rewards[2,2] = -1

        self.isObstacle[3,3] = True
        self.rewards[4,4] = 1
     

        return self.state 
    
    def getState(self):
        return self.state 
        
    def step(self, action):
        
        if self.isValidAction(action):
        
            newState = self.state_transistion_function(self.state, action)
       
            self.env[self.state] = 0
            self.env[newState] = 1

            self.state = newState

            return self.state, self.rewards[self.state], self.isTerminal[self.state]
        else:
            raise ValueError("Invalid action")
    

    def state_transistion_function(self, state, action):
        # non-deterministic
        if np.random.rand() < 0.5:
            validActionFound = False 
            while not validActionFound:
                action = np.random.choice(np.arange(self.NUM_ACTIONS))
                validActionFound, _ = self.isValidAction(action)
        
        return self.getNewState(state, action) # state = position

    def isValidAction(self, action):
            
        newState = self.getNewState(self.state, action)

        if newState == None:
            return False, None
        
        x,y = newState
        if x < 0 or y < 0 or x > 4 or y > 4:
            return False, None
        
        if self.isObstacle[x,y]:
            return False, None
        
        return True, newState

    def getNewState(self, state, action):
        x, y = state 
     
        if action == self.Actions.TOP.value:
            newState = (x - 1, y)    

        elif action == self.Actions.DOWN.value:
            newState = (x + 1, y)    

        elif action == self.Actions.RIGHT.value:
            newState = (x, y + 1)   

        elif action == self.Actions.LEFT.value:
            newState = (x, y - 1)   
        else: 
            newState = None
        
        return newState

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