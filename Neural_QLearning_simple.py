import gym
import tensorflow as tf
import numpy as np
import random

"""
Hyper Parameters
"""
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
EPSILON_DECAY_STEPS = 1000
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy
HIDDEN_NODES = 20
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Define Network Structure
state_in = tf.placeholder("float", [1, STATE_DIM])
action_in = tf.placeholder("float", [1, ACTION_DIM])  # one hot
target_in = tf.placeholder("float", [1])

h1 = tf.layers.dense(state_in, HIDDEN_NODES, activation=tf.nn.relu)
h2 = tf.layers.dense(h1, HIDDEN_NODES, activation=tf.nn.relu)
q_values = tf.layers.dense(h2, ACTION_DIM, activation=lambda x: x)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# Define Loss
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(0.0003).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

def explore(state, epsilon):
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

for episode in range(EPISODE):
    # initialize task
    state = env.reset()

    # Update epsilon once per episode - linear schedule
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)

        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        if done:
            target = reward
        else:
            target = reward + GAMMA * np.max(nextstate_q_values)

        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
        })
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                    }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                    'Average Reward:', ave_reward)

env.close()
