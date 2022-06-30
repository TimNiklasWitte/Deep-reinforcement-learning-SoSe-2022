import gym
import numpy as np
import tqdm

from Agent import *

from multiprocessing import Process, shared_memory

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


def fill_buff(num_fill_buff_threads, samples_per_thread, thread_id, agent):

    env = gym.make('LunarLander-v2')

    # Seed env and np
    env.seed(thread_id)
    np.random.seed(thread_id)

    state = env.reset()

    for i in range(samples_per_thread):
        
        action = np.random.randint(0, env.action_space.n)
        next_state, reward, done_flag, _ = env.step(action)

        idx = num_fill_buff_threads*i + thread_id
        
        agent.replay_memory.states[idx] = state
        agent.replay_memory.actions[idx] = action
        agent.replay_memory.next_states[idx] = next_state
        agent.replay_memory.rewards[idx] = reward

        agent.replay_memory.done_flags[idx] = int(done_flag)

        if done_flag:
            state = env.reset() 


def generate_samples(agent, threadId):

    if threadId == 0:
        file_path = "test_logs/test"
        train_summary_writer = tf.summary.create_file_writer(file_path)

    env = gym.make('LunarLander-v2')

    # Seed env and np
    env.seed(threadId)
    np.random.seed(threadId)

    tf.keras.backend.clear_session()

    prediction_net_1 = DQN(env.action_space.n)
    prediction_net_1.build((1, *env.observation_space.shape))

    prediction_net_2 = DQN(env.action_space.n)
    prediction_net_2.build((1, *env.observation_space.shape))

    current_prediction_net_id = agent.prediction_net_id.value

    num_episode = 0

    while True:
        
        print("here: ", agent.prediction_net_1_weights)
        if current_prediction_net_id != agent.prediction_net_id.value:
            current_prediction_net_id = agent.prediction_net_id.value
            
            if current_prediction_net_id == 1:
                weights = []
                for weight_matrix in agent.prediction_net_1_weights:
                    matrix = weight_matrix.copy()
                    weights.append(matrix)

                print(weights[0])
                prediction_net_1.set_weights(weights)

            else:
                weights = []
                for weight_matrix in agent.prediction_net_2_weights:
                    matrix = weight_matrix.copy()
                    weights.append(matrix)
                print(weights[0])

                prediction_net_2.set_weights(weights)


        done_flag = False

        
        score = 0
        cnt_steps = 0

        rewards = [] 

        state = env.reset()
        while not done_flag:

            if agent.prediction_net_id.value == 1:
                action = agent.select_action(state, prediction_net_1)
            else:
                action = agent.select_action(state, prediction_net_2)

            next_state, reward, done_flag, _ = env.step(action)

            agent.store_experience(state, action, next_state, reward, done_flag)

            state = next_state
            
            score += reward
            rewards.append(reward)
            cnt_steps += 1

        agent.strategy.reduce_epsilon()
        num_episode += 1

        if threadId == 0:
                
    
            # print(f"  Episode: {num_episode}")
            # print(f"  Epsilon: {round(agent.strategy.get_exploration_rate(), 2)}")
            # print(f"    Score: {round(score, 2)}")
            # print(f"Avg Score: {round(np.mean(rewards), 2)}")
            # print(f"    Steps: {cnt_steps}")
            # print("------------------------")
 
    
            with train_summary_writer.as_default():
                tf.summary.scalar(f"Average reward", np.mean(rewards), step=num_episode)
                tf.summary.scalar(f"Score", score, step=num_episode)
                tf.summary.scalar(f"Epsilon (EpsilonGreedyStrategy)", agent.strategy.get_exploration_rate(), step=num_episode)
                tf.summary.scalar(f"Steps per episode", cnt_steps, step=num_episode)
            


def main():


    # Init gym
    env = gym.make('LunarLander-v2')
    

    agent = Agent(input_dims=env.observation_space.shape,
                   num_actions=env.action_space.n, batch_size=32)
    # agent.q_net.summary()

  
    #
    # Fill ReplayMemory
    # 

    num_fill_buff_threads = 10
    samples_per_thread = 5000
    fill_buff_threads = []

    for thread_idx in range(num_fill_buff_threads):
        process = Process(target=fill_buff, args=(num_fill_buff_threads, samples_per_thread, thread_idx,agent))

        process.start()
        fill_buff_threads.append(process)

    for process in fill_buff_threads: 
        process.join()


    # update index of replay memory
    agent.replay_memory.idx.value = num_fill_buff_threads * samples_per_thread


    num_generate_data_threads = 1
    generate_data_threads = []

    for thread_id in range(num_generate_data_threads):
        process = Process(target=generate_samples, args=(agent, thread_id))
        process.start()
        generate_data_threads.append(process)

    agent.replay_memory.run_pre_sample_batch_threads()

    
    tf.keras.backend.clear_session()

    #
    # Training            
    # 
    
    q_net = DQN(env.action_space.n)
    q_net.build((1, *env.observation_space.shape))

    target_net = DQN(env.action_space.n)
    target_net.build((1, *env.observation_space.shape))

    agent.load(q_net)

    episode = 0
    while True:

        for i in range(1000):

            states, actions, next_state, rewards, dones = agent.replay_memory.sample_batch()

            actions = np.array(actions)
        
            predictions_currentState = q_net(states).numpy() # (64, 2)
            predictions_nextState = target_net(next_state).numpy() # (64, 2)
            best_actions = np.argmax(predictions_nextState, axis=1) # (64,)

            target = np.copy(predictions_currentState) # (64, 2)

            batch_idx = np.arange(agent.batch_size, dtype=np.int32) # (64,)
            target[batch_idx, actions] = rewards + \
            agent.gamma * predictions_nextState[batch_idx, best_actions] * (1 - dones)


            q_net.train_step(states, target)

        print("update")   
        if agent.prediction_net_id.value == 1:
            print("here 1",agent.prediction_net_1_weights)
            agent.update_prediction_net_2(q_net)
            agent.prediction_net_id.value == 2
            print(agent.prediction_net_id.value)
        else:
            agent.update_prediction_net_1(q_net)
            agent.prediction_net_id.value == 1

        
        target_net.set_weights(q_net.get_weights())

     

  


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
