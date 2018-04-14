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

REPLAY_SIZE = 10e4  # experience replay buffer size
replay_buffer = []

batch = []
BATCH_SIZE = 30

EPISODES_PER_TARGET_UPDATE = 10

# Define Network Structure
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])  # one hot
target_in = tf.placeholder("float", [None])

# Value Network
h1 = tf.layers.dense(state_in, HIDDEN_NODES, activation=tf.nn.relu)
h2 = tf.layers.dense(h1, HIDDEN_NODES, activation=tf.nn.relu)
q_values = tf.layers.dense(h2, ACTION_DIM, activation=lambda x: x)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)
network_params = tf.trainable_variables()

# Fixed Network
h1_target = tf.layers.dense(state_in, HIDDEN_NODES, activation=tf.nn.relu)
h2_target = tf.layers.dense(h1_target, HIDDEN_NODES, activation=tf.nn.relu)
q_values_target = tf.layers.dense(h2_target, ACTION_DIM, activation=lambda x: x)
target_network_params = tf.trainable_variables()[len(network_params):]

# Define Loss
loss = tf.losses.huber_loss(target_in, q_action)
optimizer = tf.train.AdamOptimizer(0.0003).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


def updateTarget():
    assert (len(target_network_params) == len(network_params))
    session.run([tf.assign(target_network_params[i], network_params[i])
                 for i in range(len(target_network_params))])


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

    if episode % EPISODES_PER_TARGET_UPDATE == 0:
        updateTarget()

    # Update epsilon once per episode - linear schedule
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > REPLAY_SIZE:
            replay_buffer.pop(0)

        if (len(replay_buffer) > BATCH_SIZE):
            batch = random.sample(replay_buffer, BATCH_SIZE)

            state_batch = [data[0] for data in batch]
            action_batch = [data[1] for data in batch]
            reward_batch = [data[2] for data in batch]
            next_state_batch = [data[3] for data in batch]
            nextstate_q_values = q_values_target.eval(feed_dict={
                state_in: next_state_batch
            })

            target_batch = []
            for i in range(0, BATCH_SIZE):
                done_batch = batch[i][4]
                if done_batch:
                    target_batch.append(reward_batch[i])
                else:
                    target_batch.append(
                        reward_batch[i] + GAMMA * np.max(nextstate_q_values[i]))

            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
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
                                                        'Average Reward:',
              ave_reward)

env.close()
