#https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np

env = gym.make('FrozenLake-v0')
# Initialize Q-table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Learning parameters
lr = 0.8
y = 0.95
num_episodes = 2000
# Create lists to contain total reward and steps every episode
rewardList = []
for i in range(num_episodes):
    state = env.reset()
    reward_all = 0
    done = False
    j = 0
    # Q-learning algorith
    while j < 99:
        j += 1
        # choose an action by greadyly (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.rand(1, env.action_space.n) * (1./(i + 1)))
        # get new state, reward
        next_state, reward, done, info = env.step(action)
        # Update Q-table
        Q[state, action] = Q[state, action] \
                           + lr * (reward + y * np.max(Q[next_state, :]) - Q[state, action])
        reward_all += reward
        state = next_state
        if done:
            break
    rewardList.append(reward_all)

print("Score over time: " + str(sum(rewardList) / num_episodes) )
print("Final Q-table values: ")
print(Q)
