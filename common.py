# coding: utf-8

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

ENV_NAME = 'VideoPinball-v0' # Gymの環境名
ENV_NAME = 'Breakout-v0' # Gymの環境名
FRAME_HEIGHT = 84 # リサイズ後のフレームの高さ
FRAME_WIDTH = 84 # リサイズ後のフレーム幅

def preprocess(observation, last_observation):
	processed_observation = np.maximum(observation, last_observation)
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255, mode='constant')

	return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT)) # 形状を合わせて状態を返す

