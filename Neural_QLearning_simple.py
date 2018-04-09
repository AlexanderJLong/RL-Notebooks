import gym
import tensorflow as tf
import numpy as np
import random
import datetime

"""
Hyper Parameters
"""
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.05  # final value of epsilon
EPSILON_DECAY_STEPS = 2000
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 30  # size of minibatch
TEST_FREQUENCY = 100  # How many episodes to run before visualizing test accuracy
SAVE_FREQUENCY = 1000  # How many episodes to run before saving model

HIDDEN_NODES = 20

ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

env = gym.make(ENV_NAME)
replay_buffer = []
time_step = 0
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n


#def dense_layer(inp, layersize, activation=tf.nn.relu):
#    """
#    Convenience function copied from Assignment 1 - Remove from final version
#    """
#    inp_size = inp.get_shape().as_list()[0]
#    w = tf.Variable(tf.truncated_normal(shape=[inp_size, layersize]))
#    b = tf.Variable(tf.constant(0.01, shape=[layersize]))
#    pre_activ = tf.matmul(w, tf.expand_dims(inp, 1), transpose_a=True) + b
#    post_activ = activation(pre_activ)
#
#    return post_activ


"""Define the neural network used to approximate the q-function

You must create, at minimum, the following tensors;
Q_values: Tensor containing Q_values for every action
Q_action: Q_value for action specified in action_in
loss: Value network is aiming to minimize
optimizer: optimizer for the network

The suggested structure is to have each output node represent a Q value for
one action. e.g. for cartpole there will be two output nodes.

Hint: Given how q-values are used within RL, is it necessary to have output
activation functions?
"""
state_in = tf.placeholder("float", [1, STATE_DIM])
action_in = tf.placeholder("float", [1, ACTION_DIM])  # one hot
target_in = tf.placeholder("float", [1])

h1 = tf.layers.dense(
    state_in,
    HIDDEN_NODES,
    activation=tf.nn.relu
)
h2 = tf.layers.dense(
    h1,
    HIDDEN_NODES,
    activation=tf.nn.relu
)

q_values = tf.layers.dense(
    h1,
    ACTION_DIM,
    activation=lambda x: x
)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(0.0003).minimize(loss)

train_cost_op = tf.summary.scalar("Training Loss", loss)


# Start session
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

# Setup Logging
logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, session.graph)

# Count the number of times we do a training batch
batch_presentations_count = 0

# Count the number of times we take a step
iterations = 0

def explore(state, epsilon):
    Q_estimates = q_values.eval(feed_dict={state_in: [state]})[0]
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    return action

for episode in range(EPISODE):
    # initialize task
    state = env.reset()

    # Update epsilon once per episode - exp decaying
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        iterations += 1
        action = explore(state, epsilon)
        one_hot_action = np.zeros(ACTION_DIM)
        one_hot_action[action] = 1
        next_state, reward, done, _ = env.step(action)

        Q_vals = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        if done:
            target = reward
        else:
            target = reward + GAMMA * np.max(Q_vals)

        summary, _ = session.run([train_cost_op, optimizer], feed_dict={
            target_in: [target],
            action_in: [one_hot_action],
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
                #env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                    })[0])
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'Batch presentations:',
              batch_presentations_count,'epsilon:', epsilon, 'Evaluation '
                                    'Average Reward:', ave_reward, )

env.close()
