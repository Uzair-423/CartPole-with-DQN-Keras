import gymnasium as gym
from tensorflow.keras.models import load_model
import numpy as np

env = gym.make('CartPole-v1', render_mode='human')
env.reset()

model = load_model('cartpoleDQN.h5')

for i in range(20):
    done = False
    state = env.reset()[0]
    while not done:
        action = np.argmax(model.predict(state.reshape([1, state.shape[0]])).flatten())
        new_state, reward, done, trunc, _ = env.step(action)
        state = new_state
    
