import numpy as np
import threading

class ReplayMemory:
    def __init__(self, capacity: int, input_dims: tuple, batch_size):
        """Init the ReplayMemory.

        Keyword arguments:
        capacity -- maximal amount of buffer entrys
        input_dims -- dimension of a game state (previous or current)
        """
        self.capacity = capacity
        self.idx = 0
        self.idx_was_overflown = False

        self.batch_size = batch_size

        # experience = state, action, next_state, reward
        self.states = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.next_states = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)

        self.done_flags = np.zeros(self.capacity, dtype=np.int32)

        #
        # Multi threading
        #
        
        self.num_sampling_threads = 15

        self.full = threading.Semaphore(0)
        self.empty = threading.Semaphore(self.num_sampling_threads)

        self.sampling_threads = []

        self.samples = []
        self.num_preSamples = 100
        self.program_terminated = False 


    def run_threads(self):
        for i in range(self.num_sampling_threads):
            thread = threading.Thread(target=self.preSample)
            thread.start()
            self.sampling_threads.append(thread)

        
    def preSample(self):
        
        while not self.program_terminated:
            
            states, actions, next_state, rewards, done_flag = self.sample_batch_singleThread()

            self.empty.acquire()
        
            self.samples.append( (states, actions, next_state, rewards, done_flag) )

            self.full.release()

     

     
    def sample_batch(self):
        
        self.full.acquire()

        states, actions, next_state, rewards, done_flag = self.samples.pop()
        
        if self.program_terminated:
            return

        self.empty.release()

        return states, actions, next_state, rewards, done_flag      

       
        

    def sample_batch_singleThread(self):
        """Samples a random batch of entry of the ReplayMemory.

        Keyword arguments:
        batch_size -- size of the batch which is sampled

        Return:
        state, action, next_state, reward 
        each of them as an np.array
        """

        if self.idx_was_overflown:
            max_mem = self.capacity
        else:
            max_mem = self.idx

        # Sampling process
        rewards = self.rewards[:max_mem]
        # Normalize between [0,1] 
        rewards_z = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
        probs = rewards_z / np.sum(rewards_z)  # sum up each value must be 1
  
        # A value shall not be sampled multiple times within a batch
        sampled_idxs = np.random.choice(max_mem, self.batch_size, replace=False, p=probs)

        states = self.states[sampled_idxs]
        actions = self.actions[sampled_idxs]
        next_state = self.next_states[sampled_idxs]
        rewards = self.rewards[sampled_idxs]

        done_flag = self.done_flags[sampled_idxs]

        return states, actions, next_state, rewards, done_flag


    def store_experience(self, state: np.array, action: int, next_state: np.array, reward: float, done_flag: bool):
        """Store a experience in the ReplayMemory.
        A experience consists of a state, an action, a next_state and a reward.

        Keyword arguments:
        state -- game state 
        action -- action taken in state
        next_state -- the new/next game state: in state do action -> next_state
        reward -- reward received
        done_flag -- does the taken action end the game?
        """

        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward

        self.done_flags[self.idx] = int(done_flag)

        self.idx += 1
        # overflow handling -> reset idx to store entries
        if self.capacity <= self.idx:
            self.idx_was_overflown = True
            self.idx = 0