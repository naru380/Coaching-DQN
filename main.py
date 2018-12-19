# coding:utf-8

import gym
import gym_tetris
import random
import numpy as np
import itertools
import csv
import datetime

from gym import wrappers
from enum import Enum

from agent.common import *
from agent.make_adviser import Agent
from agent.adviser import Adviser
from agent.player import Player


MODE = 3


class Mode(Enum):
    MAKE_ADVISER_TRAIN = 1 # Atariゲームのアドバイザ作成
    MAKE_ADVISER_TEST = 2 # 作成したAtariゲームのアドバイザの確認
    IMPLEMENT_MAIN_TASK = 3 # メインのタスク実行


def main():
    # 環境を作る
    env = gym.make(ENV_NAME)

    if MODE == Mode.MAKE_ADVISER_TRAIN.value:
        print("MODE is MAKE_ADVISER_TRAIN")

        # Agentクラスのインスタンスを作る
        agent = Agent(num_actions=env.action_space.n, load_model=False)

        for _ in range(NUM_EPISODES):
            # エピソード終了判定を初期化
            terminal = False 
            # 環境の初期化、初期画面を返す
            observation = env.reset() 
            # ランダムなフレーム数分「何もしない」行動で遷移させる
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                # 「何もしない」行動を取り、次の画面を返す
                observation, _, _, _ = env.step(0) 
            # 初期状態を作る
            state = agent.get_initial_state(observation, last_observation)
            # 1エピソードが終わるまでループ
            while not terminal: 
                last_observation = observation
                # 行動を選択    
                action = agent.get_action(state) 
                # 行動を実行して、次の画面、報酬、終了判定を返す
                observation, reward, terminal, _ = env.step(action)
                # 画面出力
                env.render() 
                # 画面の前処理
                processed_observation = preprocess(observation, last_observation) 
                # 学習を行い、次の状態を返す
                state = agent.run(state, action, reward, terminal, processed_observation) 


    elif MODE == Mode.MAKE_ADVISER_TEST.value:
        print("MODE is MAKE_ADVISER_TRAIN")

        env = wrappers.Monitor(env, directory='test', force=True)

        # Agentクラスのインスタンスを作る
        agent = Agent(num_actions=env.action_space.n, load_model=True)

        # env.monitor.start(ENV_NAME + '-test')
        for _ in range(NUM_EPISODES_AT_TEST):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)
                state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
        # env.monitor.close()


    elif MODE == Mode.IMPLEMENT_MAIN_TASK.value:
        print("MODE is IMPLEMENT_MAIN_TASK")

        logdir_path = './logdir/{0:%Y%m%d-%H:%M:%S}_{1}'.format(datetime.datetime.now(), ENV_NAME)

        if not os.path.exists(logdir_path):
            os.makedirs(logdir_path)

        f = open(logdir_path + '/log.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        labels = [
                "EPISODE",
                "TIMESTEP",
                "ADVICE_CURRENCY",
                "ACTION_CURRENCY",
                "AVERAGE_ADVICE_CURRENCY",
                "AVERAGE_ACTION_CURRENCY",
                ]

        # Adviserクラスのインスタンスを作る
        adviser = Adviser(num_actions=env.action_space.n)
        # Playerクラスのインスタンスを作る
        player = Player(num_actions=env.action_space.n, logdir_path=logdir_path)
        
        advice_count = np.zeros((adviser.num_advices, adviser.num_advices))
        action_count = np.zeros((adviser.num_actions, adviser.num_actions))

        labels.extend(["ADVISER_ADVICE" + str(i) + "PLAYER_MEAN" + str(j) for i, j in itertools.product(range(advice_count.shape[0]), range(advice_count.shape[1]))])
        labels.extend(["ADVISER_ACTION" + str(i) + "PLAYER_ACTION" + str(j) for i, j in itertools.product(range(action_count.shape[0]), range(action_count.shape[1]))])

        writer.writerow(labels)

        # タスクを開始する
        for _ in range(NUM_EPISODES):
            csvlist = []
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
                    # ゲーム画面からアドバイスを決定する
                    advice = list(adviser.get_advice(state))
                    #print("advice = {}", advice)

                # プレイヤの処理
                with player.graph.as_default():
                    # アドバイス(言語)から意味を推定する
                    mean = player.get_mean(advice)
#                   # 操作を決定する
                    action = player.get_action(state, mean)
                    #print("action = {}".format(action)))

                # 環境に対するプレイヤの行動を決定し，次のステップ(画面)へ移行する
                observation, reward, terminal, _ = env.step(action)
                env.render() # 画面出力
                processed_observation = preprocess(observation, last_observation)

                advice_count[np.argmax(advice), mean] += 1
                action_count[np.argmax(adviser.q_values.eval(feed_dict={adviser.s: [np.float32(state / 255.0)]}, session=adviser.sess)), np.argmax(player.q_values.eval(feed_dict={player.q_state: [np.float32(state / 255.0)]}, session=player.sess))
] += 1

                """
                # アドバイザの処理
                with adviser.graph.as_default():
                    # 内部状態を更新する
                    adviser.run(onehot_adviser_action, advise, reward, terminal)
                """

                # プレイヤの処理
                with player.graph.as_default():
                    # 内部状態を更新する
                    state = player.run(state, action, advice, mean, reward, terminal, processed_observation)

            csvlist.extend([player.episode, player.t])

            advice_currency = [advice_count[i, i] / np.sum(advice_count, axis=-1)[i] if np.sum(advice_count, axis=-1)[0] > 0 else 0 for i in range(advice_count.shape[-1])]
            csvlist.extend(advice_currency)
            action_currency = [action_count[i, i] / np.sum(action_count, axis=-1)[i] if np.sum(action_count, axis=-1)[0] > 0 else 0 for i in range(action_count.shape[-1])]
            csvlist.extend(action_currency)
            print("ADVICE_CURRENCY = {}".format(advice_currency))
            print("ACTION_CURRENCY = {}".format(action_currency))

            average_advice_currency = np.trace(advice_count) / np.sum(advice_count)
            csvlist.append(average_advice_currency)
            average_action_currency = np.trace(action_count) / np.sum(action_count)
            csvlist.append(average_action_currency)
            print("AVERAGE_ADVICE_CURRENCY = {}".format(average_advice_currency))
            print("AVERAGE_ACTION_CURRENCY = {}".format(average_action_currency))

            csvlist.extend([advice_count[i, j] for i, j in itertools.product(range(advice_count.shape[0]), range(advice_count.shape[0]))])
            csvlist.extend([action_count[i, j] for i, j in itertools.product(range(action_count.shape[0]), range(action_count.shape[0]))])

            writer.writerow(csvlist)

            print('')



    else:
        print("Invalid MODE is selected.")


if __name__ == '__main__':
    main()
