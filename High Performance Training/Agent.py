import numpy as np

from ReplayMemory import *
from DQN import *
from EpsilonGreedyStrategy import *


class Agent:
    def __init__(self, num_actions: int, batch_size: int, input_dims: tuple):
        """Init the Agent by creating the EpsilonGreedyStrategy, ReplayMemory
        q-network and target network. 

        Keyword arguments:
        num_actions -- Number of possible actions which can be taken in the gym.
        batch_size -- batch size, number of samples which are sampled from the replay memory during each train step
        input_dims -- dimension of a both game states (previous AND current game step concatenated)
        """
 
        self.gamma = 0.99
        self.tau = 0.01

        self.num_actions = num_actions

        self.batch_size = batch_size

        self.strategy = EpsilonGreedyStrategy(start=1.0, end=0.05, decay=0.999)
        self.replay_memory = ReplayMemory(capacity=2500000, input_dims=input_dims, batch_size=batch_size)

        #tf.keras.backend.clear_session()

        # q_net1 = DQN(num_actions)
        # q_net1.build((self.batch_size, *input_dims))

        # self.target_net = DQN(num_actions)
        # self.target_net.build((self.batch_size, *input_dims))
        # self.update_target()

        #
        # 
        #

        self.prediction_net_id = Value('i', 1)

        self.prediction_net_1_weights_shm = []
        self.prediction_net_1_weights = []

        self.prediction_net_2_weights_shm = []
        self.prediction_net_2_weights = []

        # for weight_matrix in q_net1.get_weights():
        #     size = np.prod(weight_matrix.shape)
            
        #     shm_1 = shared_memory.SharedMemory(create=True, size= size * 4)
        #     weights_matrix_1 = np.ndarray(weight_matrix.shape, dtype=np.float32, buffer=shm_1.buf)

        #     shm_2 = shared_memory.SharedMemory(create=True, size= size * 4)
        #     weights_matrix_2 = np.ndarray(weight_matrix.shape, dtype=np.float32, buffer=shm_2.buf)

        #     self.prediction_net_1_weights_shm.append(shm_1)
        #     self.prediction_net_1_weights.append(weights_matrix_1)

        #     self.prediction_net_2_weights_shm.append(shm_2)
        #     self.prediction_net_2_weights.append(weights_matrix_2)

    def load(self, q_net):

        for weight_matrix in q_net.get_weights():
            size = np.prod(weight_matrix.shape)
            
            shm_1 = shared_memory.SharedMemory(create=True, size= size * 4)
            weights_matrix_1 = np.ndarray(weight_matrix.shape, dtype=np.float32, buffer=shm_1.buf)

            shm_2 = shared_memory.SharedMemory(create=True, size= size * 4)
            weights_matrix_2 = np.ndarray(weight_matrix.shape, dtype=np.float32, buffer=shm_2.buf)

            self.prediction_net_1_weights_shm.append(shm_1)
            self.prediction_net_1_weights.append(weights_matrix_1)

            self.prediction_net_2_weights_shm.append(shm_2)
            self.prediction_net_2_weights.append(weights_matrix_2)

    def update_prediction_net_1(self, q_net):
        for idx, weight_matrix in enumerate(q_net.get_weights()):
            self.prediction_net_1_weights[idx][:] = weight_matrix.copy()
    
    def update_prediction_net_2(self, q_net):
        for idx, weight_matrix in enumerate(q_net.get_weights()):
            self.prediction_net_2_weights[idx][:] = weight_matrix.copy()

    def select_action_1(self, state: np.array):

        # Exploration
        if np.random.random() < self.strategy.get_exploration_rate():
            return np.random.randint(0, self.num_actions)
        # Exploitation
        else:
            # Add batch dim
            state = np.expand_dims(state, axis=0)

            # Select best action
            actions = self.prediction_net_1(state)
            return np.argmax(actions)
    
    def select_action_2(self, state: np.array):

        # Exploration
        if np.random.random() < self.strategy.get_exploration_rate():
            return np.random.randint(0, self.num_actions)
        # Exploitation
        else:
            # Add batch dim
            state = np.expand_dims(state, axis=0)

            # Select best action
            actions = self.prediction_net_2(state)
            return np.argmax(actions)

    def select_action(self, state: np.array, network):
        # Exploration
        if np.random.random() < self.strategy.get_exploration_rate():
            return np.random.randint(0, self.num_actions)
        # Exploitation
        else:
            # Add batch dim
            state = np.expand_dims(state, axis=0)

            # Select best action
            actions = network(state)
            return np.argmax(actions)

    def store_experience(self, state, action, next_state, reward, done):
        """Store a experience in the ReplayMemory.
        A experience consists of a state, an action, a next_state and a reward.

        Keyword arguments:
        state -- game state 
        action -- action taken in state
        next_state -- the new/next game state: in state do action -> next_state
        reward -- reward received
        done_flag -- does the taken action end the game?
        """

        self.replay_memory.store_experience(state, action, next_state, reward, done)

    def update_target(self):
        """
        The target network's weights are set to the q-network's weights
        by using Polyak averaging. 
        """

        # newWeights = self.target_net.get_weights()

        # for idx, _ in enumerate(self.q_net.get_weights()):
        #     newWeights[idx] = (1 - self.tau) * self.target_net.get_weights()[idx] + self.tau * self.q_net.get_weights()[idx]

        #print(self.target_net.get_weights()[0])
        # Polyak averaging 
        #self.target_net.set_weights(newWeights)
        self.target_net.set_weights(self.q_net.get_weights())

    def train_step(self):

        """
        A random batch is sampled from the ReplayMemory. 
        Thereafter, the q-network is trained.
        Note that, enough samples in ReplayMemory must be in the ReplayMemory. 
        Otherwise, there will be no training of the network.
        """

        # Sample a random batch
        states, actions, next_state, rewards, dones = \
            self.replay_memory.sample_batch()

        actions = np.array(actions)
       
        predictions_currentState = self.q_net(states).numpy() # (64, 2)
        predictions_nextState = self.target_net(next_state).numpy() # (64, 2)
        best_actions = np.argmax(predictions_nextState, axis=1) # (64,)

        target = np.copy(predictions_currentState) # (64, 2)

        batch_idx = np.arange(self.batch_size, dtype=np.int32) # (64,)
        target[batch_idx, actions] = rewards + \
          self.gamma * predictions_nextState[batch_idx, best_actions] * (1 - dones)
   
        #print(target[batch_idx, actions].shape) # (64,)
        #print(target[:, actions].shape) # (64, 64)

        self.q_net.train_step(states, target)