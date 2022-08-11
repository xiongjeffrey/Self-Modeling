import gym
import cv2
import numpy as np
from stable_baselines3 import A2C
import imageio

env = gym.make("gym_basic:basic-v0")
# Define and Train the agent
model = A2C('CnnPolicy', env).learn(total_timesteps=1000)


images = []
obs = model.env.reset()
# img = model.env.render(mode='rgb_array')
for i in range(1):
    # images.append(img)
    action, _ = model.predict(obs)
    print(action)
    obs, _, _ ,info = model.env.step(action)
    # print(info)
    # img = model.env.render(mode='rgb_array')


imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(info[0]['images']) if i%2 == 0], fps=2)

# image = env.im_obs()
# print("hey")
# cv2.imwrite("test.png", np.array(image))
# im = cv2.imread('test.png')
# print(im.shape)