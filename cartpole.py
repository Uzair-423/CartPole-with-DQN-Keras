import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import datetime

def agent(states, actions):
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=states, activation='relu'))
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def getQ(model,state):
    return model.predict(state.reshape([1, state.shape[0]])).flatten()

def train(replay_memory, model, model_target, done):
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    learning_rate = 0.8
    gamma = 0.95

    min_replay_size = 1000
    if len(replay_memory) < min_replay_size:
        return

    batch_size = 128
    batch = replay_memory[-batch_size:]

    states = np.array([transition[0] for transition in batch])
    Q_values = model.predict(states)

    new_states = np.array([transition[3] for transition in batch])
    new_Q_values = model_target.predict(new_states)

    X,Y = [],[]
    for index, (state, action, reward, new_state, done) in enumerate(batch):
        if not done:
            future_reward = reward + gamma * np.max(new_Q_values[index])
        else:
            future_reward = reward

        Q_values[index][action] = (1 - learning_rate) * Q_values[index][action] + learning_rate * future_reward
        X.append(state)
        Y.append(Q_values[index])

    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=2, shuffle=True)


env = gym.make('CartPole-v1')
env.reset()

n_train_episodes = 150
n_test_episodes = 10

epsilon = 1.0
min_epsilon = 0.01
decay = 0.01

replay_memory = deque(maxlen=256)
model = agent(env.observation_space.shape, env.action_space.n)
model_target = agent(env.observation_space.shape, env.action_space.n)
model_target.set_weights(model.get_weights())

target_update_counter = 0
model_update_counter = 0
ep_reward_list = []

for episode in range(n_train_episodes):
    print("Episode: ", episode)
    state = env.reset()[0]
    done = False
    ep_reward = 0

    while not done:
        target_update_counter += 1
        model_update_counter += 1

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(getQ(model,state))

        new_state, reward, done, trunc, _ = env.step(action)
        
        done = done or trunc
        ep_reward += reward

        replay_memory.append([state, action, reward, new_state, done])

        if model_update_counter == 4: train(replay_memory, model, model_target, done)
        if target_update_counter == 100: model_target.set_weights(model.get_weights())

        state = new_state
    
    print(f'Episode: {episode}, Reward: {ep_reward}, Epsilon: {epsilon}')
    ep_reward_list.append(ep_reward)
    epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-decay * episode)

env.close()
plt.plot(ep_reward_list)
plt.xlabel('Episode')
plt.ylabel('Epsiodic Reward')
plt.show()

input('Press any key to test the model...')

# Test
env = gym.make('CartPole-v1', render_mode='human')
for episode in range(n_test_episodes):
    done = False
    state = env.reset()[0]

    while not done:
        action = np.argmax(getQ(model,state))
        new_state, reward, done, trunc, _ = env.step(action)
        state = new_state

env.close()
