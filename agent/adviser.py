# coding: utf-8

from .common import *

class Adviser():
    def __init__(self, num_actions):
        self.num_actions = num_actions # 行動数
        self.num_advices = self.num_actions + NUM_ANOTHER_MEAN # アドバイス数
        self.t = 0 # タイムステップ
        self.repeated_advice = 0 # フレームスキップ間にリピートするアドバイスを保持するための変数

        # クラス専用のグラフを構築する
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            with tf.variable_scope("Adviser"):
                with tf.variable_scope("Q_Network"):
                    # Q Networkの構築
                    self.s, self.q_values, q_network = self.build_network()
                    q_network_weights = q_network.trainable_weights
                    #print(q_network_weights)

            # Sessionの構築
            self.sess = tf.Session(graph=self.graph)

            # Saverの構築
            self.saver = tf.train.Saver(q_network_weights)
            
            # 変数の初期化(Q Networkの初期化)
            self.sess.run(tf.global_variables_initializer())

            # Networkの読み込む
            self.load_network()

        # debug
        #summary_writer = tf.summary.FileWriter('data', graph=self.sess.graph)


    def build_network(self):
        # ~/.keras/keras.jsonのimage_data_formatを'channel_last'から'channel_first'に変更
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), name='Conv2D_1'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', name='Conv2D_2'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='Conv2D_3'))
        model.add(Flatten(name='Flatten'))
        model.add(Dense(512, activation='relu', name='Dense_1'))
        model.add(Dense(self.num_actions, name='Dense_2'))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT], name='state')
        q_values = model(s)

        return s, q_values, model


    def load_network(self):
        K.set_session(self.sess)

        checkpoint = tf.train.get_checkpoint_state(TRAINED_ADVISER_NETWORK_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')


    def get_action(self, state):
        K.set_session(self.sess)
        action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess))
        
        return action


    def get_advice(self, state, player_action):
        K.set_session(self.sess)

        advice = self.repeated_advice
        
        if self.t % ACTION_INTERVAL == 0:
            adviser_action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess))
            
            if random.random() > 0.5:
                advice = adviser_action + NUM_ANOTHER_MEAN
            else:
                if adviser_action == player_action:
                    advice = AnotherMean.EVALUATE_GOOD.value
                else:
                    advice = AnotherMean.EVALUATE_BAD.value

            self.repeated_advice = advice

        self.t += 1
        
        # 言語としてone_hot_vectorを返す
        return np.identity(self.num_advices)[advice]
