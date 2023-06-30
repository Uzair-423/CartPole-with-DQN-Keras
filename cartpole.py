import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import datetime
import random

def agent(states, actions):
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=states, activation='relu'))
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.008), metrics=['accuracy'])
    return model

def getQ(model,state):
    return model.predict(state.reshape([1, state.shape[0]])).flatten()

def train(replay_memory, model, model_target, done):
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    learning_rate = 0.9
    gamma = 0.95

    min_replay_size = 1000
    if len(replay_memory) < min_replay_size:
        return

    batch_size = 128
    batch = random.sample(replay_memory, batch_size)
    # batch = replay_memory[-batch_size:]

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

def validate(model):
    env1 = gym.make('CartPole-v1', render_mode='human')
    for i in range(5):
        state = env1.reset()[0]
        done = False
        while not done:
            action = np.argmax(getQ(model,state))
            new_state, reward, done, trunc, _ = env1.step(action)
            state = new_state

    env1.close()

env = gym.make('CartPole-v1')
env.reset()

n_train_episodes = 160
n_test_episodes = 10

epsilon = 1.0
min_epsilon = 0.01
decay = 0.01

replay_memory = deque(maxlen=1024)
model = agent(env.observation_space.shape, env.action_space.n)
model_target = agent(env.observation_space.shape, env.action_space.n)
model_target.set_weights(model.get_weights())

target_update_counter = 0
model_update_counter = 0
ep_reward_list = []
avg_reward_list = []

for episode in range(n_train_episodes):
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

        if model_update_counter == 4: 
            train(replay_memory, model, model_target, done)
            model_update_counter = 0
        if target_update_counter == 100: 
            model_target.set_weights(model.get_weights())
            target_update_counter = 0
            

        state = new_state
    
    print(f'Episode: {episode}, Reward: {ep_reward}, Epsilon: {epsilon}')
    ep_reward_list.append(ep_reward)
    if episode<10:
        avg_reward_list.append(ep_reward)
    else:
        avg_reward_list.append(np.mean(ep_reward_list[-10:]))
    epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-decay * episode)

    if episode == n_train_episodes//2 or episode == n_train_episodes//4*3:
        validate(model)

env.close()
plt.plot(ep_reward_list, label='Episode Reward')
plt.plot(avg_reward_list, label='Average Reward')
plt.xlabel('Episode')
plt.ylabel('Epsiodic Reward')
plt.legend()
plt.show()

input('Press Enter to test the model...')

# Test
eval_env = gym.make('CartPole-v1', render_mode='human')
for episode in range(n_test_episodes):
    done = False
    state = eval_env.reset()[0]

    while not done:
        action = np.argmax(getQ(model,state))
        new_state, reward, done, trunc, _ = eval_env.step(action)
        state = new_state

eval_env.close()

# Save the model
prompt = input('Do you want to save the model? (y/n): ')
if prompt == 'y':
    model.save('cartpoleDQN.h5')
