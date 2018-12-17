# coding: utf-8

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from time import sleep
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import concatenate
from keras.optimizers import SGD, Adam
from keras import backend as K
from enum import Enum

KERAS_BACKEND = 'tensorflow'

ENV_NAME = 'Tetris-v0' # Gymの環境名
FRAME_HEIGHT = 84 # リサイズ後のフレームの高さ
FRAME_WIDTH = 84 # リサイズ後のフレーム幅
NUM_EPISODES = 10000000 # プレイするエピソード数
STATE_LENGTH = 4 # 状態を構成するフレーム数
GAMMA = 0.99 # 割引率
EXPLORATION_STEPS = 100000 # ε-greedey法のεが減少していくフレーム数
INITIAL_EPSILON = 1.0 # ε-greedy法のεの初期値
FINAL_EPSILON = 0.1 # ε-greedy法のεの終値
INITIAL_REPLAY_SIZE = 20000 # 学習前に事前確保するReplay Memory数
NUM_REPLAY_MEMORY = 400000 # Replay Memory数
BATCH_SIZE = 32 # バッチサイズ
TARGET_UPDATE_INTERVAL = 10000 # Target Networkの更新をする間隔
ACTION_INTERVAL = STATE_LENGTH # フレームスキップ数
TRAIN_INTERVAL = STATE_LENGTH # 学習を行なう間隔
LEARNING_RATE = 0.00025 # RMSPropで使われる学習率
MOMENTUM = 0.25 # RSMPropで使われるモメンタム
MIN_GRAD = 0.01 # RSMPropで使われる0で割るのを防ぐための値
SAVE_INTERVAL = 300000  # Networkを保存する間隔
NO_OP_STEPS = 30 # エピソード開始時に「何もしない」最大フレーム数（初期状態をランダムにする）
TRAINED_ADVISER_NETWORK_PATH = 'trained_adviser/saved_networks/' + ENV_NAME # 学習済みのアドバイザのQ_Netowrkの重みの保存場所
TRAINED_ADVISER_SUMMARY_PATH = 'trained_adviser/summary/' + ENV_NAME # アドバイザ学習時のデータの保存場所
SAVE_NETWORK_PATH = 'log/saved_networks/' + ENV_NAME # タスク実行時のQ_Networkの重みを保存する場所
SAVE_SUMMARY_PATH = 'log/summary/' + ENV_NAME # タスク実行時の学習データを保存する場所
NUM_EPISODES_AT_TEST = 30  # テストプレイで実行するエピソード数



class AnotherMean(Enum):
    NOOP = 0 # 何もしない

NUM_ANOTHER_MEAN = len(list(AnotherMean))
NUM_ANOTHER_MEAN =  0



def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255, mode='constant')

    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT)) # 形状を合わせて状態を返す


class Debug:
    def __init__(self, model):
        self.model = model
        self.gradients = self.get_gradients()

    def get_gradients(self):
        outputTensor = self.model.get_output_at(0)
        listOfVariableTensors = self.model.trainable_weights
        gradients = K.gradients(outputTensor, listOfVariableTensors)
        
        return gradients

    def evaluate_gradients(self, sess, input_data):
        evaluated_gradients = sess.run(self.gradients, feed_dict={self.model.get_input_at(0): input_data})

        return evaluated_gradients
