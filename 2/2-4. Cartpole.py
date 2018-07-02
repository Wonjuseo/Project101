# Cart Pole example

import gym
# Environment
env = gym.make('CartPole-v0')
env.reset()
# Parameters
random_episodes = 0
reward_sum = 0


while random_episodes < 10:
    # Rendering
    env.render()
    # Get action
    action = env.action_space.sample()
    # Update state, reward, done
    observation, reward, done, _ = env.step(action)
    print(observation,reward,done)

    # Add reward
    reward_sum += reward

    # if it fails, the results were shown
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
