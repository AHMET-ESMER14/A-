import gym
import numpy as np
import tensorflow as tf

env = gym.make('LunarLander-v2',render_mode = "human")


model = tf.keras.models.load_model("C:/Users/Monster/Desktop/LunarLander/3000_trained_model.h5")

def render_model(model, env, episodes=5):
    for episode in range(episodes):
        observation , _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            
            observation = np.expand_dims(observation, axis=0) # diziyi yatay hale getiriyor
        
            action_probs = model.predict(observation)[0]
            
            action = np.argmax(action_probs)
            
            next_observation, reward, done, _, __ = env.step(action)
            
            env.render()
            
            observation = next_observation
            
            total_reward += reward
        
        print("Episode:", episode+1, "Total Reward:", total_reward)
    
    env.close()

render_model(model, env)