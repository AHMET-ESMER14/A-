import gym, random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



class DQLAgent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9993
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.memory = deque(maxlen=4000)
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, s):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(s)
        return np.argmax(act_values[0])

    def replay(self,batch_size):
        if len(agent.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        #minibatch = [np.resize(array, (32, 5)) for array in minibatch]
        #minibatch = np.array(minibatch)############################
        minibatch = np.array(minibatch, dtype=object)
        not_done_indices = np.where(minibatch[:, 4] == False) 
        y = np.copy(minibatch[:, 2])

        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
            
            y[not_done_indices] += np.multiply(self.gamma, predict_sprime_target[not_done_indices, np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(batch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())
            
if __name__ == "__main__":
    env = gym.make('MountainCar-v0',render_mode = "human")

    agent = DQLAgent(env)
    state_number = env.observation_space.shape[0]
    
    batch_size = 32
    episodes = 3000
    rewards_per_episode = []

    
    for e in range(episodes):
        
        state, _ = env.reset()
        #state_array = state[0]
        #state = np.reshape(state_array, [1, state_number])
        #state = np.reshape(state,[1,env.observation_space.shape[0]])
        state = np.reshape(state, [1, state_number])

        total_reward = 0
        for time in range(400):
            
            #env.render()###############################################################

            action = agent.act(state)
            
            next_state, reward, done, _ ,__ = env.step(action)
            next_state = np.reshape(next_state, [1, state_number])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            agent.replay(batch_size)################################################################

            total_reward += reward
            
            if done:
                agent.targetModelUpdate()
                break

        rewards_per_episode.append(total_reward)
        agent.adaptiveEGreedy()

        print('Episode: {}, Reward: {}'.format(e,total_reward))
        if((e%30) == 0):
            agent.model.save("C:/Users/Monster/Desktop/CartPole/mountain_Car/deneme.h5")
            print("V_Model kaydedildi.")   
            

    agent.model.save("C:/Users/Monster/Desktop/CartPole/mountain_Car/deneme2.h5")
    print("Model kaydedildi.")    

    env.close()
    # Eğitim tamamlandıktan sonra ödül grafiğini çizin
    plt.plot(rewards_per_episode)
    plt.title('3000 Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig("C:/Users/Monster/Desktop/CartPole/mountain_Car/V_3000_rewards_plot.png")
    plt.show()    
    
#%% test
"""
import time

trained_model = agent
state, _ = env.reset()

state = np.reshape(state, [1, env.observation_space.shape[0]])
while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _ , __ = env.step(action)
    next_state = np.reshape(next_state, [1,env.observation_space.shape[0]])
    state = next_state
    #time.sleep(0.1)
    if done:
        env.close()
        break
print("Done")    
"""