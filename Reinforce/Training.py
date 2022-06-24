import gym
import numpy as np

from PolicyNet import *

def main():
    
    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    env = gym.make('CartPole-v0')

    policy_net = PolicyNet(env.action_space.n)
    policy_net.build((1, *env.observation_space.shape))

    with train_summary_writer.as_default():
    
        for num_episode in range(5000):

            state = env.reset()

            states = [state]
            rewards = []
            score = 0
            for num_steps in range(1000):

                state = tf.expand_dims(state, axis=0)
                actions = policy_net(state)
                best_action = np.argmax(actions, axis=-1)[0]
            

                next_state, reward, done_flag, _ = env.step(best_action)

                state = next_state

                states.append(state)
                rewards.append(reward)
                score += reward
                if done_flag:
                    break 
            
            tf.summary.scalar(f"Steps", num_steps, step=num_episode)
            tf.summary.scalar(f"Average reward", np.mean(rewards), step=num_episode)
            tf.summary.scalar(f"Score", score, step=num_episode)

            gamma = 0.99
            returns = []
        
            for j in range(len(rewards)):

                g_t = 0
                for i in range(len(rewards[j:])):
                    g_t += (gamma**i) * rewards[i]

                returns.append(g_t)
        

            for idx, state in enumerate(states[:-1]):

                state = tf.expand_dims(state, axis=0)

                policy_net.train_step(state, returns[idx])
    
   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
