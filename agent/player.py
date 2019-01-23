# coding: utf-8

from .common import *

class Player():
    def __init__(self, num_actions, logdir_path):
        self.num_actions = num_actions # 行動数
        self.num_advices = NUM_ANOTHER_MEAN # アドバイス数
        self.epsilon = INITIAL_EPSILON # ε-greedy法のεの初期化
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS # εの減少率
        self.t = 0 # タイムステップ
        self.repeated_action = 0 # フレームスキップ間にリピートする行動を保持するための変数
        self.repeated_mean = 0 # フレームスキップ間にリピートする意味を保持するための変数
        
        # Replay Memoryの構築
        self.replay_memory = deque()

        # summaryに使用するパラメータ
        self.total_clipped_reward = 0
        self.total_non_clipped_reward = 0
        self.total_adviced_reward = 0
        self.action_net_total_q_max = 0
        self.action_net_total_loss = 0
        self.mean_net_total_q_max = 0
        self.mean_net_total_loss = 0
        self.duration = 0
        self.episode = 0

        # クラス専用のグラフを構築する
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            with tf.variable_scope("Player"):
                # ゲーム画面から直接行動を学習するネットワークの構築
                with tf.variable_scope("Action_Network"):
                    with tf.variable_scope("Q_Network"):
                        # Q Networkの構築
                        self.q_state, self.action_q_values, action_q_network = self.build_action_network()
                        action_q_network_weights = action_q_network.trainable_weights

                    with tf.variable_scope("Target_Network"):
                        # Target Networkの構築
                        self.target_state, self.action_target_q_values, action_target_network = self.build_action_network()
                        action_target_network_weights = action_target_network.trainable_weights

                    # 定期的にTarget Networkを更新するための処理の構築
                    self.update_action_target_network = [action_target_network_weights[i].assign(action_q_network_weights[i]) for i in range(len(action_target_network_weights))]

                    # 誤差関数や最適化のための処理の構築
                    self.action, self.action_teacher_signal, self.action_net_loss, self.action_net_grad_update = self.build_action_net_training_op(action_q_network_weights)

                # アドバイスから意味推定を学習するネットワークの構築
                with tf.variable_scope("Mean_Network"):
                    with tf.variable_scope("Q_Network"):
                        # Q Networkの構築
                        self.q_advice, self.mean_q_values, self.mean_q_network = self.build_mean_network()
                        mean_q_network_weights = self.mean_q_network.trainable_weights

                    with tf.variable_scope("Target_Network"):
                        # Target Networkの構築
                        self.target_advice, self.mean_target_q_values, mean_target_network = self.build_mean_network()
                        mean_target_network_weights = mean_target_network.trainable_weights

                    # 定期的にTarget Networkを更新するための処理の構築
                    self.update_mean_target_network = [mean_target_network_weights[i].assign(mean_q_network_weights[i]) for i in range(len(mean_target_network_weights))]

                    # 誤差関数や最適化のための処理の構築
                    self.mean, self.mean_teacher_signal, self.mean_net_loss, self.mean_net_grad_update = self.build_mean_net_training_op(mean_q_network_weights)

            # Sessionの構築
            self.sess = tf.Session(graph=self.graph)
            
            self.action_q_net_saver = tf.train.Saver(action_q_network_weights, name='Action_Network_Saver')
            self.mean_q_net_saver = tf.train.Saver(mean_q_network_weights, name='Mean_Network_Saver')

            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            self.summary_logdir_path = logdir_path + SAVE_SUMMARY_PATH
            self.summary_writer = tf.summary.FileWriter(self.summary_logdir_path, self.sess.graph)
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

            self.network_logdir_path = logdir_path + SAVE_NETWORK_PATH
            if not os.path.exists(self.network_logdir_path):
                os.makedirs(self.network_logdir_path)

            # 変数の初期化(Q Networkの初期化)
            self.sess.run(tf.global_variables_initializer())

            # Target Networkの初期化
            self.sess.run(self.update_action_target_network)
            self.sess.run(self.update_mean_target_network)


    def build_action_network(self):
        # ~/.keras/keras.jsonのimage_data_formatを'channel_last'から'channel_first'に変更
        display_input = Input(shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=tf.float32)
        x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='normal', name='Conv2D_1')(display_input)
        x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='normal', name='Conv2D_2')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='normal', name='Conv2D_3')(x)
        x = Flatten(name='Flatten')(x)
        x = Dense(512, activation='relu', kernel_initializer='normal', name='Dense_1')(x)
        x = Dense(self.num_actions, kernel_initializer='normal', name='Dense_2')(x)
        model = Model(inputs=[display_input], outputs=x)

        state = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(inputs=[state])

        return state, q_values, model


    def build_action_net_training_op(self, q_network_weights):
        action = tf.placeholder(tf.int64, [None], name='Action') # 行動
        teacher_signal = tf.placeholder(tf.float32, [None], name='Teacher_Signal') # 教師信号

        with tf.variable_scope('1-Hot_Vecor_Generator'):
            action_one_hot = tf.one_hot(action, self.num_actions, 1.0, 0.0) # 行動をone hot vectorに変換する
        with tf.variable_scope('Q_Value_Calculator'):
            q_value = tf.reduce_sum(tf.multiply(self.action_q_values, action_one_hot), reduction_indices=1) # 行動のQ値を計算

        with tf.variable_scope('Error_Function'):
            # エラークリップ(the loss is quadratic when the error is in (-1, 1), and linear outside of that region） = Humber Lossに相当？
            error = tf.abs(teacher_signal - q_value) # 最大値と最小値を指定する
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0) 
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part) # 誤差関数

        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD) # 最適化手法を定義
            grad_update = optimizer.minimize(loss, var_list=q_network_weights) # 誤差最小化

        return action, teacher_signal, loss, grad_update

    
    def build_mean_network(self):
        advice_input = Input(shape=(self.num_advices, ))
        x = Dense(self.num_advices*2, activation='sigmoid', kernel_initializer='uniform', name='Dense_1')(advice_input)
        x = Dense(self.num_advices, activation='sigmoid', kernel_initializer='uniform', name='Dense_2')(x)
        model = Model(inputs=[advice_input], outputs=x)
        advice = tf.placeholder(tf.float32, [None, self.num_advices], name='Advice')
        q_values = model(inputs=[advice])

        self.debug = Debug(model)

        return advice, q_values, model


    def build_mean_net_training_op(self, q_network_weights):
        mean = tf.placeholder(tf.int64, [None], name='Mean') # 意味
        teacher_signal = tf.placeholder(tf.float32, [None], name='Teacher_Signal') # 教師信号

        with tf.variable_scope('1-Hot_Vecor_Generator'):
            mean_one_hot = tf.one_hot(mean, self.num_advices, 1.0, 0.0) #意味をone hot vectorに変換する
        with tf.variable_scope('Q_Value_Calculator'):
            q_value = tf.reduce_sum(tf.multiply(self.mean_q_values, mean_one_hot), reduction_indices=1) # 行動のQ値を計算

        with tf.variable_scope('Error_Function'):
            # エラークリップ(the loss is quadratic when the error is in (-1, 1), and linear outside of that region） = Humber Lossに相当？
            error = tf.abs(teacher_signal - q_value) # 最大値と最小値を指定する
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0) 
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part) # 誤差関数

        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.RMSPropOptimizer(MEAN_NET_LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD) # 最適化手法を定義
            grad_update = optimizer.minimize(loss, var_list=q_network_weights) # 誤差最小化

        return mean, teacher_signal, loss, grad_update


    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation) # 現在の画面と前画面の各ピクセルごとに最大値を取る
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255, mode='constant') # グレイスケールに変換後、メモリを圧迫しないようにunsigned char型に変換する
        state = [processed_observation for _ in range(STATE_LENGTH)] # フレームをスキップする分、状態を複製する
        return np.stack(state, axis=0) # 複製した状態を連結して返す


#    def get_mean(self, advice):
#        mean = self.repeated_mean # 意味をリピート
#
#        K.set_session(self.sess)
#        
#        if AQUIRED_LANGUAGE == True:
#            if self.t % ACTION_INTERVAL == 0:
#                if np.argmax(advice) == AnotherMean.EVALUATE_GOOD.value:
#                    mean = 0
#                else:
#                    mean = 1
#                self.repeated_mean = mean # フレームスキップ間にリピートする意味を格納
#        else:
#            if self.t % ACTION_INTERVAL == 0:
#                """
#                if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
#                    mean = random.randrange(self.num_advices)
#                else:
#                    mean = np.argmax(self.mean_q_values.eval(feed_dict={self.q_advice: [advice]}, session=self.sess))
#                """
#                mean = np.argmax(self.mean_q_values.eval(feed_dict={self.q_advice: [advice]}, session=self.sess))
#                self.repeated_mean = mean # フレームスキップ間にリピートする意味を格納
#
#        return mean


    def get_mean(self, advice):
        mean = self.repeated_mean # 意味をリピート

        K.set_session(self.sess)
        
        if AQUIRED_LANGUAGE == True:
            if self.t % ACTION_INTERVAL == 0:
                if np.argmax(advice) == AnotherMean.EVALUATE_GOOD.value:
                    mean = 0
                else:
                    mean = 1
                self.repeated_mean = mean # フレームスキップ間にリピートする意味を格納
        else:
            if self.t % ACTION_INTERVAL == 0:
                if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
                    mean = random.randrange(self.num_advices)
                else:
                    mean = np.argmax(self.mean_q_values.eval(feed_dict={self.q_advice: [advice]}, session=self.sess))
                #mean = np.argmax(self.mean_q_values.eval(feed_dict={self.q_advice: [advice]}, session=self.sess))
                self.repeated_mean = mean # フレームスキップ間にリピートする意味を格納

        return mean


    def get_action(self, state):
        action = self.repeated_action # 行動をリピート

        K.set_session(self.sess)

        if self.t % ACTION_INTERVAL == 0:
            if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.action_q_values.eval(feed_dict={self.q_state: [np.float32(state / 255.0)]}, session=self.sess))
            self.repeated_action = action # フレームスキップ間にリピートする行動を格納

        # εを線形に減少させる
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action


    def run(self, state, action, advice, mean, reward, terminal, observation):
        # プレイヤー用のセッションに切り替える
        K.set_session(self.sess)

        # 次の状態を作成
        next_state = np.append(state[1:, :, :], observation, axis=0)
        #print(next_state)

        self.total_non_clipped_reward += reward

        # 報酬を固定、正は1、負は−1、0はそのまま
        reward = np.sign(reward)

        # 評価から報酬を得る
        evaluation_reward = 0
        if mean == AnotherMean.EVALUATE_GOOD.value:
            evaluation_reward = 1
        elif mean == AnotherMean.EVALUATE_BAD.value:
            evaluation_reward = -1
        else:
            pass

        # Replay Memoryに遷移を保存
        self.replay_memory.append((state, action, advice, mean, reward, evaluation_reward, next_state, terminal))
        
        # Replay Memoryが一定数を超えたら、古い遷移から削除
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Q Networkの学習
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Target Networkの更新
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_action_target_network, options=self.run_options, run_metadata=self.run_metadata)
                self.sess.run(self.update_mean_target_network, options=self.run_options, run_metadata=self.run_metadata)

            # Networkの保存
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.action_q_net_saver.save(self.sess, self.network_logdir_path + '/' + 'Player_Action_Network', global_step=(self.t))
                save_path = self.mean_q_net_saver.save(self.sess, self.network_logdir_path + '/' + 'Player_Mean_Network', global_step=(self.t))
                print('Successfully saved: ' + save_path)
                """
                f_replay_memory = open(self.network_logdir_path + '/replay_memory', 'wb')
                pickle.dump(list(self.replay_memory), f_replay_memory)
                f_replay_memory.close
                """
                f_parameters = open(self.network_logdir_path + '/parameters', 'wb')
                pickle.dump([self.t, self.episode, self.epsilon], f_parameters)
                f_parameters.close

        self.total_clipped_reward += reward
        self.total_adviced_reward += reward + evaluation_reward
        self.action_net_total_q_max += np.max(self.action_q_values.eval(feed_dict={self.q_state: [np.float32(state / 255.0)]}, session=self.sess))
        self.mean_net_total_q_max += np.max(self.mean_q_values.eval(feed_dict={self.q_advice: [advice]}, session=self.sess))
        self.duration += 1

        if terminal:
            # summaryの書き込み
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [
                        float(self.duration),
                        self.total_non_clipped_reward,
                        self.total_clipped_reward,
                        self.total_adviced_reward,
                        self.action_net_total_q_max / float(self.duration),
                        self.action_net_total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),
                        self.mean_net_total_q_max / float(self.duration),
                        self.mean_net_total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),
                        ]

                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i]),
                        }, options=self.run_options, run_metadata=self.run_metadata)
                summary_str = self.sess.run(self.summary_op, options=self.run_options, run_metadata=self.run_metadata)
                self.summary_writer.add_run_metadata(self.run_metadata, 'step%03d' % (self.episode + 1))
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_NON-CLIPPED_REWARD: {4:3.0f} / TOTAL_CLIPPED_REWARD: {5:3.0f} / ACTION_NET_AVG_MAX_Q: {6:2.4f} / ACTION_NET_AVG_LOSS: {7:.5f} / MEAN_NET_AVG_MAX_Q: {8:2.4f} / MEAN_NET_AVG_LOSS: {9:.5f} / MODE: {10}'.format(
                self.episode + 1, 
                self.t, 
                self.duration, 
                self.epsilon,
                self.total_non_clipped_reward, self.total_clipped_reward,
                self.action_net_total_q_max / float(self.duration),
                self.action_net_total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),
                self.mean_net_total_q_max / float(self.duration),
                self.mean_net_total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),
                mode))

            self.duration = 0
            self.total_clipped_reward = 0
            self.total_non_clipped_reward = 0
            self.total_adviced_reward = 0
            self.action_net_total_q_max = 0
            self.action_net_total_loss = 0
            self.mean_net_total_q_max = 0
            self.mean_net_total_loss = 0
            self.episode += 1

        self.t += 1 # タイムステップ

        return next_state


    def train_network(self):
        state_batch = []
        action_batch = []
        advice_batch = []
        mean_batch = []
        reward_batch = []
        evaluation_reward_batch = []
        next_state_batch = []
        terminal_batch = []
        action_net_teacher_signal_batch = []
        mean_net_teacher_signal_batch = []

        # Replay Memoryからランダムにミニバッチをサンプル
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            advice_batch.append(data[2])
            mean_batch.append(data[3])
            reward_batch.append(data[4])
            evaluation_reward_batch.append(data[5])
            next_state_batch.append(data[6])
            terminal_batch.append(data[7])

        K.set_session(self.sess)

        # 終了判定をTrueは1に、Falseは0に変換
        terminal_batch = np.array(terminal_batch) + 0
        # Target Networkで次の状態でのQ値を計算
        action_target_q_values_batch = self.action_target_q_values.eval(feed_dict={self.target_state: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess) 
        #mean_target_q_values_batch = self.mean_target_q_values.eval(feed_dict={self.target_advice: np.float32(np.array(advice_batch))}, session=self.sess) 
        # 教師信号を計算
        action_net_teacher_signal_batch = reward_batch + evaluation_reward_batch + (1 - terminal_batch) * GAMMA * np.max(action_target_q_values_batch, axis=1)
        #mean_net_teacher_signal_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(mean_target_q_values_batch, axis=1)
        mean_net_teacher_signal_batch = reward_batch

        # 勾配法による誤差最小化
        action_net_loss, _ = self.sess.run([self.action_net_loss, self.action_net_grad_update], feed_dict={
            self.q_state: np.float32(np.array(state_batch) / 255.0),
            self.action: action_batch,
            self.action_teacher_signal: action_net_teacher_signal_batch
            }, options=self.run_options, run_metadata=self.run_metadata)

        mean_net_loss, _ = self.sess.run([self.mean_net_loss, self.mean_net_grad_update], feed_dict={
            self.q_advice: np.float32(np.array(advice_batch)),
            self.mean: mean_batch,
            self.mean_teacher_signal: mean_net_teacher_signal_batch
            }, options=self.run_options, run_metadata=self.run_metadata)

        self.action_net_total_loss += action_net_loss
        self.mean_net_total_loss += mean_net_loss


    def train_action_network(self):
        state_batch = []
        action_batch = []
        #advice_batch = []
        #mean_batch = []
        reward_batch = []
        evaluation_reward_batch = []
        next_state_batch = []
        terminal_batch = []
        action_net_teacher_signal_batch = []
        #mean_net_teacher_signal_batch = []

        # Replay Memoryからランダムにミニバッチをサンプル
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            #advice_batch.append(data[2])
            #mean_batch.append(data[3])
            reward_batch.append(data[4])
            evaluation_reward_batch.append(data[5])
            next_state_batch.append(data[6])
            terminal_batch.append(data[7])

        K.set_session(self.sess)

        # 終了判定をTrueは1に、Falseは0に変換
        terminal_batch = np.array(terminal_batch) + 0
        # Target Networkで次の状態でのQ値を計算
        action_target_q_values_batch = self.action_target_q_values.eval(feed_dict={self.target_state: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess) 
        #mean_target_q_values_batch = self.mean_target_q_values.eval(feed_dict={self.target_advice: np.float32(np.array(advice_batch))}, session=self.sess) 
        # 教師信号を計算
        action_net_teacher_signal_batch = reward_batch + evaluation_reward_batch + (1 - terminal_batch) * GAMMA * np.max(action_target_q_values_batch, axis=1)
        #mean_net_teacher_signal_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(mean_target_q_values_batch, axis=1)

        # 勾配法による誤差最小化
        action_net_loss, _ = self.sess.run([self.action_net_loss, self.action_net_grad_update], feed_dict={
            self.q_state: np.float32(np.array(state_batch) / 255.0),
            self.action: action_batch,
            self.action_teacher_signal: action_net_teacher_signal_batch
            }, options=self.run_options, run_metadata=self.run_metadata)

        self.action_net_total_loss += action_net_loss


    def train_mean_network(self):
        #state_batch = []
        #action_batch = []
        advice_batch = []
        mean_batch = []
        reward_batch = []
        #evaluation_reward_batch = []
        #next_state_batch = []
        #terminal_batch = []
        #action_net_teacher_signal_batch = []
        mean_net_teacher_signal_batch = []

        # Replay Memoryからランダムにミニバッチをサンプル
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            #state_batch.append(data[0])
            #action_batch.append(data[1])
            advice_batch.append(data[2])
            mean_batch.append(data[3])
            reward_batch.append(data[4])
            #evaluation_reward_batch.append(data[5])
            #next_state_batch.append(data[6])
            #terminal_batch.append(data[7])

        K.set_session(self.sess)

        # 終了判定をTrueは1に、Falseは0に変換
        #terminal_batch = np.array(terminal_batch) + 0
        # Target Networkで次の状態でのQ値を計算
        #mean_target_q_values_batch = self.mean_target_q_values.eval(feed_dict={self.target_advice: np.float32(np.array(advice_batch))}, session=self.sess) 
        # 教師信号を計算
        #mean_net_teacher_signal_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(mean_target_q_values_batch, axis=1)
        mean_net_teacher_signal_batch = reward_batch
        
        # 勾配法による誤差最小化
        mean_net_loss, _ = self.sess.run([self.mean_net_loss, self.mean_net_grad_update], feed_dict={
            self.q_advice: np.float32(np.array(advice_batch)),
            self.mean: mean_batch,
            self.mean_teacher_signal: mean_net_teacher_signal_batch
            }, options=self.run_options, run_metadata=self.run_metadata)

        self.mean_net_total_loss += mean_net_loss


    def setup_summary(self):
        K.set_session(self.sess)

        with tf.variable_scope('Summary'):
            episode_duration = tf.Variable(0., name='Episode')
            tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)

            episode_total_non_clipped_reward = tf.Variable(0., name='Total_Non-clinpped_Reward')
            tf.summary.scalar(ENV_NAME + '/Total Non-Clipped Reward/Episode', episode_total_non_clipped_reward)

            episode_total_clipped_reward = tf.Variable(0., name='Total_Clipped_Reward')
            tf.summary.scalar(ENV_NAME + '/Total Clipped Reward/Episode', episode_total_clipped_reward)

            episode_total_adviced_reward = tf.Variable(0., name='Total_Adviced_Reward')
            tf.summary.scalar(ENV_NAME + '/Total Adviced Reward/Episode', episode_total_adviced_reward)

            episode_action_net_avg_max_q = tf.Variable(0., name='Avg_Action_Net_Max_Q')
            tf.summary.scalar(ENV_NAME + '/Average Action Network Max Q/Episode', episode_action_net_avg_max_q)

            episode_action_net_avg_loss = tf.Variable(0., name='Avg_Action_Net_Loss')
            tf.summary.scalar(ENV_NAME + '/Average Action Network_Loss/Episode', episode_action_net_avg_loss)

            episode_mean_net_avg_max_q = tf.Variable(0., name='Avg_Mean_Net_Max_Q')
            tf.summary.scalar(ENV_NAME + '/Average Mean Network Max_Q/Episode', episode_mean_net_avg_max_q)

            episode_mean_net_avg_loss = tf.Variable(0., name='Avg_Mean_Net_Loss')
            tf.summary.scalar(ENV_NAME + '/Average Mean Network Loss/Episode', episode_mean_net_avg_loss)

            summary_vars = [
                    episode_duration, 
                    episode_total_non_clipped_reward, 
                    episode_total_clipped_reward, 
                    episode_total_adviced_reward,
                    episode_action_net_avg_max_q, 
                    episode_action_net_avg_loss, 
                    episode_mean_net_avg_max_q, 
                    episode_mean_net_avg_loss
                    ]

            summary_placeholders = [tf.placeholder(tf.float32, name='Summary_Placeholder_' + str(i)) for i in range(len(summary_vars))]
            update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
            summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op

    
    def get_network_outputs(self, state, advice):
        K.set_session(self.sess)

        action = np.argmax(self.action_q_values.eval(feed_dict={self.q_state: [np.float32(state / 255.0)]}, session=self.sess))
        mean = np.argmax(self.mean_q_values.eval(feed_dict={self.q_advice: [advice]}, session=self.sess))

        return action, mean
