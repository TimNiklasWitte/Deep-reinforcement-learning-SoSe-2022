import gym
import numpy as np
import tqdm

from Agent import *



def main():
    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_episods = 5000 # after init replay memory
    update = 100

    # Init gym
    env = gym.make('LunarLander-v2')
 
    agent = Agent(input_dims=env.observation_space.shape,
                   num_actions=env.action_space.n, batch_size=32)
    agent.q_net.summary()

    #
    # Fill ReplayMemory
    # 
    print("Fill ReplayMemory:")

    state = env.reset()

    # 500000
    for _ in tqdm.tqdm(range(500000), position=0, leave=True):

        action = agent.select_action(state)
        next_state, reward, done_flag, _ = env.step(action)

        agent.store_experience(state, action, next_state, reward, done_flag)
        
        if done_flag:
            state = env.reset() 
    

    agent.replay_memory.run_threads()

    #
    # Training            
    # 
    with train_summary_writer.as_default():

        for episode in range(num_episods):

            done_flag = False

            score = 0  # sum of rewards
            rewards = []

            cnt_steps = 0
            state = env.reset()
            while not done_flag:
                action = agent.select_action(state)
                next_state, reward, done_flag, _ = env.step(action)

                agent.store_experience(state, action, next_state, reward, done_flag)

                state = next_state
                agent.train_step()

                score += reward

                rewards.append(reward)
                cnt_steps += 1

            # Reduce epsilon after each episode
            agent.strategy.reduce_epsilon()

            # Update target network
            if episode % update == 0:
                agent.update_target()

            # Save weights
            if episode % 50 == 0:
                agent.q_net.save_weights(f"./saved_models/trainied_weights_epoch_{episode}", save_format="tf")

            tf.summary.scalar(f"Average reward", np.mean(rewards), step=episode)
            tf.summary.scalar(f"Score", score, step=episode)
            tf.summary.scalar(f"Epsilon (EpsilonGreedyStrategy)", agent.strategy.get_exploration_rate(), step=episode)
            tf.summary.scalar(f"Steps per episode", cnt_steps, step=episode)

            print(f"  Episode: {episode}")
            print(f"  Epsilon: {round(agent.strategy.get_exploration_rate(), 2)}")
            print(f"    Score: {round(score, 2)}")
            print(f"Avg Score: {round(np.mean(rewards), 2)}")
            print(f"    Steps: {cnt_steps}")
            print("------------------------")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
