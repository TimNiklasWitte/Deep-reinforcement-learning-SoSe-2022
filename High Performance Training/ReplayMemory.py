import numpy as np
from multiprocessing import Process, Value, Semaphore, shared_memory, Pool
from time import sleep


class ReplayMemory:
    def __init__(self, capacity: int, input_dims: tuple, batch_size):
        """Init the ReplayMemory.

        Keyword arguments:
        capacity -- maximal amount of buffer entrys
        input_dims -- dimension of a game state (previous or current)
        """
        self.capacity = capacity
      
        self.idx = Value('i', 0)

        self.idx_was_overflown = Value('i', 0)

        self.batch_size = batch_size

        # experience = state, action, next_state, reward
        self.states = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.next_states = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)

        self.done_flags = np.zeros(self.capacity, dtype=np.int32)
        

        #
        #
        #
        sum_dims_input = np.prod(*input_dims)
        self.states_shm = shared_memory.SharedMemory(create=True, size=self.capacity * sum_dims_input * 4)
        self.actions_shm = shared_memory.SharedMemory(create=True, size=self.capacity * 4)
        self.next_states_shm = shared_memory.SharedMemory(create=True, size=self.capacity * sum_dims_input * 4)
        self.rewards_shm = shared_memory.SharedMemory(create=True, size=self.capacity * 4)
        self.done_flags_shm = shared_memory.SharedMemory(create=True, size=self.capacity * 4)


        self.states = np.ndarray((self.capacity, *input_dims), dtype=np.float32, buffer=self.states_shm.buf)
        self.actions = np.ndarray((self.capacity,), dtype=np.int32, buffer=self.actions_shm.buf)
        self.next_states = np.ndarray((self.capacity, *input_dims), dtype=np.float32, buffer=self.next_states_shm.buf)
        self.rewards = np.ndarray((self.capacity,), dtype=np.float32, buffer=self.rewards_shm.buf)
        self.done_flags = np.ndarray((self.capacity,), dtype=np.int32, buffer=self.done_flags_shm.buf)

       
        #
        # Presampling
        #
        
        self.num_pre_sampling_batches_threads = 5
        self.pre_sampling_batches_threads = []

        self.num_pre_sampled_batches = 200

        self.program_terminated = False 

    
        self.pre_sampled_batches_states_shm = shared_memory.SharedMemory(create=True, size=self.num_pre_sampled_batches * self.batch_size * sum_dims_input * 4)
        self.pre_sampled_batches_actions_shm = shared_memory.SharedMemory(create=True, size=self.num_pre_sampled_batches * self.batch_size * 4)
        self.pre_sampled_batches_next_states_shm = shared_memory.SharedMemory(create=True, size=self.num_pre_sampled_batches * self.batch_size * sum_dims_input * 4)
        self.pre_sampled_batches_rewards_shm = shared_memory.SharedMemory(create=True, size=self.num_pre_sampled_batches * self.batch_size * 4)
        self.pre_sampled_batches_done_flags_shm = shared_memory.SharedMemory(create=True, size=self.num_pre_sampled_batches * self.batch_size * 4)
        
        self.pre_sampled_batches_states = np.ndarray((self.num_pre_sampled_batches, self.batch_size, *input_dims), dtype=np.float32, buffer=self.pre_sampled_batches_states_shm.buf)
        self.pre_sampled_batches_actions = np.ndarray((self.num_pre_sampled_batches, self.batch_size,), dtype=np.int32, buffer=self.pre_sampled_batches_actions_shm.buf)
        self.pre_sampled_batches_next_states = np.ndarray((self.num_pre_sampled_batches, self.batch_size, *input_dims), dtype=np.float32, buffer=self.pre_sampled_batches_next_states_shm.buf)
        self.pre_sampled_batches_rewards = np.ndarray((self.num_pre_sampled_batches, self.batch_size,), dtype=np.float32, buffer=self.pre_sampled_batches_rewards_shm.buf)
        self.pre_sampled_batches_done_flags = np.ndarray((self.num_pre_sampled_batches, self.batch_size,), dtype=np.int32, buffer=self.pre_sampled_batches_done_flags_shm.buf)

        self.num_current_pre_sampled_batches = Value('i', 0)


    def run_pre_sample_batch_threads(self):
        for i in range(self.num_pre_sampling_batches_threads):
            thread = Process(target=self.pre_sample_batch)
            thread.start()
            self.pre_sampling_batches_threads.append(thread)
        
       
        
    def pre_sample_batch(self):
        
        while True:
            
            states, actions, next_states, rewards, done_flags = self.sample_batch_singleThread()

    
            idx = self.num_current_pre_sampled_batches.value 
            
            while self.num_pre_sampled_batches <= idx:
                idx = self.num_current_pre_sampled_batches.value 
                self.num_current_pre_sampled_batches.value += 1
                sleep(0.1)


            self.pre_sampled_batches_states[idx] = states
            self.pre_sampled_batches_actions[idx] = actions
            self.pre_sampled_batches_next_states[idx] = next_states
            self.pre_sampled_batches_rewards[idx] = rewards
            self.pre_sampled_batches_done_flags[idx] = done_flags

            


     
    def sample_batch(self):
        
        
        idx = self.num_current_pre_sampled_batches.value 

        while idx == 0:
            idx = self.num_current_pre_sampled_batches.value 
            self.num_current_pre_sampled_batches.value -= 1

        states = self.pre_sampled_batches_states[idx]
        actions = self.pre_sampled_batches_actions[idx]
        next_state = self.pre_sampled_batches_next_states[idx]
        rewards = self.pre_sampled_batches_rewards[idx]
        done_flags = self.pre_sampled_batches_done_flags[idx]

         
        return states, actions, next_state, rewards, done_flags     

       
        

    def sample_batch_singleThread(self):
        """Samples a random batch of entry of the ReplayMemory.

        Keyword arguments:
        batch_size -- size of the batch which is sampled

        Return:
        state, action, next_state, reward 
        each of them as an np.array
        """

        if self.idx_was_overflown.value == 1:
            max_mem = self.capacity
        else:
            max_mem = self.idx.value

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

        done_flags = self.done_flags[sampled_idxs]

        return states, actions, next_state, rewards, done_flags


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

        current_idx = self.idx.value

        self.states[current_idx] = state
        self.actions[current_idx] = action
        self.next_states[current_idx] = next_state
        self.rewards[current_idx] = reward

        self.done_flags[current_idx] = int(done_flag)
        
        self.idx.value += 1
       
        # overflow handling -> reset idx to store entries
        if self.capacity <= self.idx.value:
            self.idx_was_overflown.value = True
            self.idx.value = 0
        