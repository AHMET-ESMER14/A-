# lunar_lander_play.py

import gym
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt



def show_image(image):
    plt.imshow(image)
    plt.show()



def main():
    env = gym.make('LunarLander-v2')
    model = load_model("C:/Users/Monster/Desktop/LunarLander/trained_model.h5")  # Eğitilmiş modeli yükleyin

    for i in range(5):  # 5 oyun denemesi yapın
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            image = env.render()
            #show_image(image)
            normalized_image = image.astype(float) / 255.0

            #Görüntüyü gösterin
            show_image(normalized_image)
            
            #state = np.array(state)  # Giriş verilerini NumPy dizisine dönüştürün
            #action = np.argmax(model.predict(state)[0]) 

#            action = np.argmax(model.predict(np.array(state))[0])
            action = np.argmax(model.predict(np.array([state]))[0])  # Modeli kullanarak bir aksiyon seçin
            next_state, reward, done, _ , __ = env.step(action)  # Seçilen aksiyonu ortama uygulayın
            total_reward += reward
            state = next_state

        print(f"Oyun {i+1}, Toplam Ödül: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()