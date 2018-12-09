# coding:utf-8

import gym
import random
import numpy as np

from gym import wrappers
from enum import Enum

from agent.common import *
from agent.make_adviser import Agent
from agent.adviser import Adviser
from agent.player import Player

class Mode(Enum):
    MAKE_ADVISER_TRAIN = 1 # アドバイザ作成
    MAKE_ADVISER_TEST = 2 # 作成したアドバイザの確認
    IMPLEMENT_MAIN_TASK = 3 # メインのタスク実行

MODE = Mode.MAKE_ADVISER_TRAIN
MODE = Mode.MAKE_ADVISER_TEST
MODE = Mode.IMPLEMENT_MAIN_TASK



def main():
    # 環境を作る
    env = gym.make(ENV_NAME)

    if MODE == Mode.MAKE_ADVISER_TRAIN:
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


    elif MODE == Mode.MAKE_ADVISER_TEST:
        print("MODE is MALE_ADVISER_TRAIN")

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


    elif MODE == Mode.IMPLEMENT_MAIN_TASK:
        print("MODE is IMPLEMENT_MAIN_TASK")

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
                    #adviser_action = adviser.get_action(state)
                    advice = list(adviser.get_advice(state))
                    #print("advice = {}", advice)
                    # 操作をアドバイス(言語)に変換する
                    #onehot_adviser_action = [np.identity(env.action_space.n)[adviser_action]]
                    #advise = adviser.get_advise_from_action(onehot_adviser_action)
                    #one_hot_advice = [np.identity(env.action_space.n + 1)[advice]]
                    #print("one_hot_advice = {}".format(one_hot_advice))

                    #print("adviser:{}".format(np.argmax(advice)))

                    #print("one_hot_action = \n{}".format(np.identity(env.action_space.n)[adviser_action]))
                    #print("advise = {}".format(advise))

                # プレイヤの処理
                with player.graph.as_default():
                    # アドバイス(言語)を操作に変換する
#                    # 操作を決定する
                    mean = player.get_mean(advice)
                    action = player.get_action(state, mean)
                    #print("advised_action = {}".format(np.argmax(advised_action)))
                    # ゲーム画面から操作を決定する
                    #player_action = player.get_action(state, advise)
                    #print(player_action)

                # 環境に対するプレイヤの行動を決定し，次のステップ(画面)へ移行する
                observation, reward, terminal, _ = env.step(action)
                env.render() # 画面出力
                processed_observation = preprocess(observation, last_observation)

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

            intention0 = 0
            intention1 = 1
            intention2 = 2
            intention3 = 3
            #intention4 = 4
            """
            advise1 =  adviser.get_advice([np.identity(env.action_space.n+1)[0]])
            advise2 =  adviser.get_advice([np.identity(env.action_space.n+1)[1]])
            advise3 =  adviser.get_advice([np.identity(env.action_space.n+1)[2]])
            advise4 =  adviser.get_advisce([np.identity(env.action_space.n)[3]])
            mean0 = np.argmax(player.get_action_from_advise(advise1))
            advised_action2 = np.argmax(player.get_action_from_advise(advise2))
            advised_action3 = np.argmax(player.get_action_from_advise(advise3))
            advised_action4 = np.argmax(player.get_action_from_advise(advise4))
            """
            advice0 = np.identity(env.action_space.n+NUM_ANOTHER_MEAN)[intention0]
            advice1 = np.identity(env.action_space.n+NUM_ANOTHER_MEAN)[intention1]
            advice2 = np.identity(env.action_space.n+NUM_ANOTHER_MEAN)[intention2]
            advice3 = np.identity(env.action_space.n+NUM_ANOTHER_MEAN)[intention3]
            
            #advice4 = np.identity(env.action_space.n+1)[intention4]
            mean0 = player.debug_mean(advice0)
            mean1 = player.debug_mean(advice1)
            mean2 = player.debug_mean(advice2)
            mean3 = player.debug_mean(advice3)
            #mean4 = player.get_mean(advice4)

            #print("adviser: [{}, {}, {}, {}, {}]".format(intention0, intention1, intention2, intention3, intention4))
            #print("player : [{}, {}, {}, {}, {}]".format(mean0, mean1, mean2, mean3, mean4))
            print("adviser: [{}, {}, {}, {}]".format(intention0, intention1, intention2, intention3))
            print("player : [{}, {}, {}, {}]".format(mean0, mean1, mean2, mean3))
    else:
        print("Invalid MODE is selected.")


if __name__ == '__main__':
    main()
