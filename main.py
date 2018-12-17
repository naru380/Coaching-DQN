# coding:utf-8

import readchar
import gym_tetris
import random
import numpy as np
import threading

from gym import wrappers
from enum import Enum
from time import sleep

from agent.common import *
from agent.make_adviser_on_tetris import AgentOnTetris
from agent.adviser_on_tetris import AdviserOnTetris
from agent.player_on_tetris import PlayerOnTetris


MODE = 1
HUMAN_PLAY = False

class Mode(Enum):
    MAKE_ADVISER_TRAIN_ON_TETRIS = 1 # テトリスのアドバイザの作成
    MAKE_ADVISER_TEST_ON_TETRIS = 2 # 作成したテトリスのアドバイザの確認
    IMPLEMENT_MAIN_TASK_ON_TETRIS = 3 # メインのタスク実行


class Key(Enum):
    A = 'a' # 何もしない
    S = 's' # 左に移動
    D = 'd' # 右に移動
    F = 'f' # 下に移動
    G = 'g' # 反時計回りに回転
    H = 'h' # 時計回りに回転
    """
    J = 'j' # 左下に移動
    K = 'k' # 右下に移動
    L = 'l' # 反時計回りに回転しながら左に移動
    SEMICOLON = ';' # 反時計回りに回転しながら右に移動
    COLON = ':' # 時計回りに回転しながら左に移動
    RIGHTSQUAREBRACKET = ']' # 時計回りに回転しながら右に移動
    """

PRESSED_KEY = ''
RECIEVE_FLAG = False
def keyboard_monitor():
    global PRESSED_KEY
    #PRESSED_KEY = readchar.readchar()
    if HUMAN_PLAY:
        PRESSED_KEY = readchar.readkey()
    while not RECIEVE_FLAG:
        pass



def main():
    # 環境を作る
    env = gym_tetris.make(ENV_NAME)

    new_action_space_n = len(Key)

    if MODE == Mode.MAKE_ADVISER_TRAIN_ON_TETRIS.value:
        print("MODE is MAKE_ADVISER_TRAIN_ON_TETRIS")

        # テトリス学習用Agentクラスのインスタンスを作る
        agent = AgentOnTetris(num_actions=new_action_space_n, load_model=False)

        global PRESSED_KEY
        global RECIEVE_FLAG
        PRESSED_KEY = ''
        
        # キーボード監視用のスレッドを立てる
        get_key = threading.Thread(target=keyboard_monitor)
        get_key.setDaemon(True)
        get_key.start()
        
        end_flag = False
        human_mode = False # 初期状態では自動で学習する
        speed = 0.0
        for _ in range(NUM_EPISODES):
            terminal = False
            human_reward = 0
            observation = env.reset()
            last_observation = observation
            state = agent.get_initial_state(observation, last_observation)

            while not terminal:
                sleep(speed)
                if PRESSED_KEY == '': # 何も押されなかったら場合は何も行わない
                    human_action = 0
                else:
                    RECIEVE_FLAG = True
                    if PRESSED_KEY == 'q': # q: 終了
                        end_flag = True
                        break
                    elif PRESSED_KEY == '/': # /: 報酬+1
                        human_reward = +1
                    elif PRESSED_KEY == '\\': # \: 報酬-1
                        human_reward = -1
                    elif PRESSED_KEY == 'm': # m: 自動or人間操作の切り替え
                        human_mode = not human_mode
                    elif PRESSED_KEY == 'o': # o: スピードダウン
                        speed += 0.01
                    elif PRESSED_KEY == 'p': # p: スピードアップ
                        speed -= 0.01
                        if speed <= 0.0:
                            speed = 0
                    else:
                        for i, key in enumerate(Key):
                            if PRESSED_KEY == key.value: # 人間操作時の各操作
                                human_action = i 
                                break

                last_observation = observation
                agent_action = agent.get_action(state)

                if human_mode:
                    action = human_action
                else:
                    action = agent_action

                observation, reward, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = agent.run(state, action, reward+human_reward, terminal, processed_observation)

                if RECIEVE_FLAG:
                    PRESSED_KEY = ''
                    RECIEVE_FLAG = False
                    human_reward = 0
                    get_key = threading.Thread(target=keyboard_monitor)
                    get_key.setDaemon(True)
                    get_key.start()

            if end_flag:
                break


    elif MODE == Mode.MAKE_ADVISER_TEST_ON_TETRIS.value:
        print("MODE is MAKE_ADVISER_TRAIN_ON_TETRIS")

        env = wrappers.Monitor(env, directory='test', force=True)

        # Agentクラスのインスタンスを作る
        agent = AgentOnTetris(num_actions=new_action_space_n, load_model=True)

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


    elif MODE == Mode.IMPLEMENT_MAIN_TASK_ON_TETRIS.value:
        print("MODE is IMPLEMENT_MAIN_TASK_ON_TETRIS")

        # Adviserクラスのインスタンスを作る
        adviser = Adviser(num_actions=new_action_space_n)
        # Playerクラスのインスタンスを作る
        player = Player(num_actions=new_action_space_n)

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
            advise1 =  adviser.get_advice([np.identity(new_action_space_n+1)[0]])
            advise2 =  adviser.get_advice([np.identity(new_action_space_n+1)[1]])
            advise3 =  adviser.get_advice([np.identity(new_action_space_n+1)[2]])
            advise4 =  adviser.get_advisce([np.identity(new_action_space_n)[3]])
            mean0 = np.argmax(player.get_action_from_advise(advise1))
            advised_action2 = np.argmax(player.get_action_from_advise(advise2))
            advised_action3 = np.argmax(player.get_action_from_advise(advise3))
            advised_action4 = np.argmax(player.get_action_from_advise(advise4))
            """
            advice0 = np.identity(new_action_space_n+NUM_ANOTHER_MEAN)[intention0]
            advice1 = np.identity(new_action_space_n+NUM_ANOTHER_MEAN)[intention1]
            advice2 = np.identity(new_action_space_n+NUM_ANOTHER_MEAN)[intention2]
            advice3 = np.identity(new_action_space_n+NUM_ANOTHER_MEAN)[intention3]
            
            #advice4 = np.identity(new_action_space_n+1)[intention4]
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
