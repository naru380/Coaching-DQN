# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf

from agent.common import *
from agent.adviser import Adviser
from agent.player import Player


def main():
	# Breakout-v0の環境を作る
	env = gym.make(ENV_NAME)

	# Adviserクラスのインスタンスを作る
	adviser = Adviser(num_actions=env.action_space.n)
	# Playerクラスのインスタンスを作る
	player = Player(num_actions=env.action_space.n)

	# タスクを開始する
	for _ in range(NUM_EPISODES):
		terminal = False
		observation = env.reset()

		# エピソードの開始時にランダムな回数行動し，エピソード毎の初期状態を変化させる
		for _ in range(random.randint(1, NO_OP_STEPS)):
			last_observation = observation
			observation, _, _, _ = env.step(0)

		# プレイヤの処理
		with player.graph.as_default():
			# ゲーム画面から初期状態を得る
			state = player.get_initial_state(observation, last_observation)

		while not terminal:
			last_observation = observation

			# アドバイザの処理
			with adviser.graph.as_default():
				# ゲーム画面から操作を決定する
				adviser_action = adviser.get_action(state)
				print(adviser_action)
				# 操作をアドバイス(言語)に変換する
				advise = adviser.get_advise_from_action([np.identity(env.action_space.n)[adviser_action]])
				print("one_hot_action = \n{}".format(np.identity(env.action_space.n)[adviser_action]))
				print("advise = {}".format(advise))

			# プレイヤの処理
			with player.graph.as_default():
				# アドバイス(言語)を操作に変換する
				advised_action = player.get_action_from_advise(advise)
				print("advised_action = {}".format(np.argmax(advised_action)))
				# ゲーム画面から操作を決定する
				player_action = player.get_action(state)

			# 環境に対するプレイヤの行動を決定し，次のステップ(画面)へ移行する
			observation, reward, terminal, _ = env.step(adviser_action)
			env.render() # 画面出力
			processed_observation = preprocess(observation, last_observation)

			# プレイヤの処理
			with player.graph.as_default():
				# 内部状態を更新する
				state = player.run(state, player_action, reward, terminal, processed_observation)


if __name__ == '__main__':
	main()
