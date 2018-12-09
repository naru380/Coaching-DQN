# coding: utf-8

from .common import *

class Adviser():
    def __init__(self, num_actions):
        self.num_actions = num_actions # 行動数
        self.t = 0 # タイムステップ
        # self.repeated_action = 0 # フレームスキップ間にリピートする行動を保持するための変数
        self.repeated_advice = 0 # フレームスキップ間にリピートするアドバイスを保持するための変数

        """
        self.total_reward = 0
        self.average_total_reward = 0
        self.episode = 1
        self.advise_memory = np.array([[]])
        self.action_memory = np.array([[]])
        """

        # クラス専用のグラフを構築する
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            with tf.variable_scope("Adviser"):
                with tf.variable_scope("Q_Network"):
                    # Q Networkの構築
                    self.s, self.q_values, q_network = self.build_network()
                    q_network_weights = q_network.trainable_weights
                    #print(q_network_weights)

                """
                # Language Networkの構築
                with tf.variable_scope("Langage_Network"):
                    self.la, self.advise, self.language_network = self.build_language_network()
                    language_network_weights = self.language_network.trainable_weights
                    
                # Language Networkの誤差関数や最適化のための処理の構築
                #self.ln_a, self.ln_y, self.ln_loss, self.ln_grad_update = self.build_training_op(language_network_weights)
                """

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
        checkpoint = tf.train.get_checkpoint_state(TRAINED_ADVISER_NETWORK_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')


    """
    def get_action(self, state):
        action = self.repeated_action

        if self.t % ACTION_INTERVAL == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess))
            self.repeated_action = action

        self.t += 1

        return action
    """


    """
    def _get_advice(self, state):
        advice = self.repeated_advice
        
        if self.t % ACTION_INTERVAL == 0:
            if random.random() <= 0.25:
                advice = AnotherMean.NOOP.value
            else:
                advice = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess)) + NUM_ANOTHER_MEAN
            self.repeated_advice = advice

        self.t += 1
        
        # 言語としてone_hot_vectorを返す
        return np.identity(self.num_actions+NUM_ANOTHER_MEAN)[advice]
    """


    def get_advice(self, state):
        advice = self.repeated_advice
        
        if self.t % ACTION_INTERVAL == 0:
            advice = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess))
            self.repeated_advice = advice

        self.t += 1
        
        # 言語としてone_hot_vectorを返す
        return np.identity(self.num_actions)[advice]

    
    """
    def build_language_network(self):
        model = Sequential()
        model.add(Dense(units=1, activation='sigmoid', use_bias=False, input_dim=self.num_actions, kernel_initializer='normal'))
        #model.add(Dense(units=1, activation='sigmoid', init='normal'))
        #model.add(Dense(units=1, activation='sigmoid', input_dim=4))
        a = tf.placeholder(tf.float32, shape=(1, self.num_actions), name='action')
        advise = model(a)

        #model.compile(loss='categorical_crossentropy',  optimizer='rmsprop', metrics=['accuracy'])
        #model.compile(loss='mean_squared_error',  optimizer='rmsprop', metrics=['accuracy'])
        model.compile(loss='mean_squared_error',  optimizer=SGD(lr=0.1), metrics=['accuracy'])
        #model.compile(loss='sparse_categorical_crossentropy',  optimizer='rmsprop', metrics=['accuracy'])
        #model.compile(loss='binary_crossentropy',  optimizer=Adam(lr=1), metrics=['accuracy'])

        self.debug_LN = Debug(model)

        return a, advise, model


    def get_advise_from_action(self, action):
        advise = self.advise.eval(feed_dict={self.la: np.array(action)}, session=self.sess)

        return advise
    """


    """
    def run(self, action, advise, reward, terminal):
        rand = random.uniform(-1, 1)

        if advise + rand >= 1:
            teacher_signal = np.array([[0.9999999]])
        elif advise + rand < 0:
            teacher_signal = np.array([[0.0]])
        else:
            teacher_signal = advise + rand
        
        #teacher_signal = [[0.3]]
        #action = [1,0,0,0]
        self.action_memory = np.append(self.action_memory, action)
        self.advise_memory = np.append(self.advise_memory, teacher_signal)

        #sleep(1)

        if terminal == False:
            self.total_reward += reward
        else:
            if self.total_reward < self.average_total_reward:
                # オンライン学習
                #self.language_network.fit(np.array(action), teacher_signal, verbose=1)
                # バッチ学習
                self.train_language_network()
            self.average_total_reward = (self.average_total_reward*(self.episode-1) + self.total_reward) / self.episode
            self.total_reward = 0
            self.episode += 1
            self.advise_memory = np.array([[]])
            self.action_memory = np.array([[]])


    def train_language_network(self):
        #print("weights before lerning is \n{}".format(self.language_network.get_weights()))
        #print(self.get_advise_from_action([[1,0,0,0]]))
        #print(self.get_advise_from_action([[0,1,0,0]]))
        #print(self.get_advise_from_action([[0,0,1,0]]))
        #print(self.get_advise_from_action([[0,0,0,1]]))
        K.set_session(self.sess)
        self.language_network.fit(np.reshape(self.action_memory, (-1, self.num_actions)), np.reshape(self.advise_memory, (-1, 1)), epochs=1, verbose=1)
        #print(self.language_network.trainable_weights)
        #print(self.language_network.summary())
        #print("weights after lerning is \n{}".format(self.language_network.get_weights()))
        #print(self.get_advise_from_action([[1,0,0,0]]))
        #print(self.get_advise_from_action([[0,1,0,0]]))
        #print(self.get_advise_from_action([[0,0,1,0]]))
        #print(self.get_advise_from_action([[0,0,0,1]]))
        #print(self.debug_LN.evaluate_gradients(self.sess, [[1,1,1,1]]))
    """
