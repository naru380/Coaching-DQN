# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense

import common
import agent

ENV_NAME = common.ENV_NAME # Gymの環境名
NUM_EPISODES = 12000 # プレイするエピソード数
NO_OP_STEPS = 30 # エピソード開始時に「何もしない」最大フレーム数（初期状態をランダムにする）
LOAD_NETWORK = True 


def main():
	# Breakout-v0の環境を作る
	env = gym.make(ENV_NAME)
	# Agentクラスのインスタンスを作る
	adviser = agent.Adviser(num_actions=env.action_space.n)
	player = agent.Player(num_actions=env.action_space.n)

	for _ in range(NUM_EPISODES):
		terminal = False
		observation = env.reset()
		for _ in range(random.randint(1, NO_OP_STEPS)):
			last_observation = observation
			observation, _, _, _ = env.step(0)
		state = player.get_initial_state(observation, last_observation)
		while not terminal:
			last_observation = observation
			adviser_action = adviser.get_action(state)
			player_action = player.get_action(state)
			print("adviser_action={}, player_action={}".format(adviser_action, player_action))
			# action に関する処理
			observation, reward, terminal, _ = env.step(adviser_action)
			# アドバイザのプレイヤがアクションを実行してからしか知ることができない
			if adviser_action == player_action:
				print("good")
			else:
				print("bad")
			processed_observation = common.preprocess(observation, last_observation)
			state = player.run(state, player_action, reward, terminal, processed_observation)
			
			
			terminal = True
	
	
	'''
	if TRAIN:  # Train mode
		for _ in range(NUM_EPISODES):
			terminal = False # エピソード終了判定を初期化
			observation = env.reset() # 環境の初期化、初期画面を返す
			# ランダムなフレーム数分「何もしない」行動で遷移させる
			for _ in range(random.randint(1, NO_OP_STEPS)):
				last_observation = observation
				observation, _, _, _ = env.step(0)  # 「何もしない」行動を取り、次の画面を返す
			state = agent.get_initial_state(observation, last_observation) # 初期状態を作る
			# 1エピソードが終わるまでループ
			while not terminal: 
				last_observation = observation
				action = agent.get_action(state) # 行動を選択
				observation, reward, terminal, _ = env.step(action) # 行動を実行して、次の画面、報酬、終了判定を返す
				# env.render() # 画面出力
				processed_observation = preprocess(observation, last_observation) # 画面の前処理
				state = agent.run(state, action, reward, terminal, processed_observation) # 学習を行い、次の状態を返す
	else:  # Test mode
		# env.monitor.start(ENV_NAME + '-test')
		for _ in range(NUM_EPISODES_AT_TEST):
			terminal = False
			observation = env.reset()
			for _ in range(random.randint(1, NO_OP_STEPS)):
				last_observation = observation
				observation, _, _, _ = env.step(0)  # Do nothing
			state = agent.get_initial_state(observation, last_observation)
			while not terminal:
				last_observation = observation
				action = agent.get_action_at_test(state)
				observation, , terminal, _ = env.step(action)
				env.render()
				processed_observation = preprocess(observation, last_observation)
				state = np.append(state[1:, :, :], processed_observation, axis=0)
		# env.monitor.close()
	'''

if __name__ == '__main__':
	main()
