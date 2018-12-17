# coding:utf-8

from .common import *

#INITIAL_REPLAY_SIZE_ON_TETRIS = 1000 # 学習前に事前確保するReplay Memory数
INITIAL_REPLAY_SIZE_ON_TETRIS = 50000 # 学習前に事前確保するReplay Memory数(本家実装)
#TARGET_UPDATE_INTERVAL_ON_TETRIS = 500 # Target Networkの更新をする間隔
TARGET_UPDATE_INTERVAL_ON_TETRIS = 10000 # Target Networkの更新をする間隔(本家実装)
SAVE_INTERVAL_ON_TETRIS = 10000  # Networkを保存する間隔
#NITIAL_EPSILON_ON_TETRIS = 0.1 # ε-greedy法のεの初期値

#GAMMA_ON_TETRIS = 0.8 # 割引率
GAMMA_ON_TETRIS = 0.99 # 割引率



class AgentOnTetris():
    def __init__(self, num_actions, load_model):
        self.num_actions = num_actions # 行動数
        self.epsilon = INITIAL_EPSILON # ε-greedy法のεの初期化
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS # εの減少率
        self.t = 0 # タイムステップ
        self.repeated_action = 0 # フレームスキップ間にリピートする行動を保持するための変数

        #summaryに使用するパラメータ
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # クラス専用のグラフを構築する
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            with tf.variable_scope('Adviser'):
                with tf.variable_scope('Replay_Memory'):
                    # Replay Memoryの構築
                    self.replay_memory = deque()

                with tf.variable_scope('Q_Network'):
                    # Q Networkの構築
                    self.s, self.q_values, q_network = self.build_network()
                    q_network_weights = q_network.trainable_weights

                with tf.variable_scope('Target_Network'):
                    # Target Networkの構築
                    self.st, self.target_q_values, target_network = self.build_network()
                    target_network_weights = target_network.trainable_weights

                # 定期的にTarget Networkを更新するための処理の構築
                with tf.variable_scope('Copy_Network_Weight'):
                    self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

                # 誤差関数や最適化のための処理の構築
                self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)
            
            # Sessionの構築
            self.sess = tf.InteractiveSession()
            #self.sess = tf.Session(graph=self.graph)
            
            self.saver = tf.train.Saver(q_network_weights)
            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            self.summary_writer = tf.summary.FileWriter(TRAINED_ADVISER_SUMMARY_PATH, self.sess.graph)

            if not os.path.exists(TRAINED_ADVISER_NETWORK_PATH):
                os.makedirs(TRAINED_ADVISER_NETWORK_PATH)


            # 変数の初期化(Q Networkの初期化)
            self.sess.run(tf.global_variables_initializer())

            # Networkの読み込む
            if load_model:
                self.load_network()

            # Target Networkの初期化
            self.sess.run(self.update_target_network)


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


    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None], name='Action') # 行動
        y = tf.placeholder(tf.float32, [None], name='Teacher_Signal') # 教師信号

        with tf.variable_scope('1-Hot_Vecor_Generator'):
            a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0) # 行動をone hot vectorに変換する
        with tf.variable_scope('Q_Value_Calculator'):
            q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1) # 行動のQ値を計算

        with tf.variable_scope('Error_Function'):
            # エラークリップ(the loss is quadratic when the error is in (-1, 1), and linear outside of that region） = Humber Lossに相当？
            error = tf.abs(y - q_value) # 最大値と最小値を指定する
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0) 
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part) # 誤差関数

        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD) # 最適化手法を定義
            grad_update = optimizer.minimize(loss, var_list=q_network_weights) # 誤差最小化

        return a, y, loss, grad_update


    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation) # 現在の画面と前画面の各ピクセルごとに最大値を取る
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255, mode='constant') # グレイスケールに変換後、メモリを圧迫しないようにunsigned char型に変換する
        state = [processed_observation for _ in range(STATE_LENGTH)] # フレームをスキップする分、状態を複製する
        return np.stack(state, axis=0) # 複製した状態を連結して返す


    def get_action(self, state):
        action = self.repeated_action # 行動をリピート

        if self.t % ACTION_INTERVAL == 0:
            if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE_ON_TETRIS:
                action = random.randrange(self.num_actions) # ランダムに行動を選択
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            #self.repeated_action = action # フレームスキップ間にリピートする行動を格納
            self.repeated_action = 0 # テトリスの場合、例えば回転が4回リピートされると1周してしまうため、リピート期間は何も行わない。

        # εを線形に減少させる
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE_ON_TETRIS:
            self.epsilon -= self.epsilon_step

        return action


    def run(self, state, action, reward, terminal, observation):
        # 次の状態を作成
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # 報酬を固定、正は1、負は−1、0はそのまま
        reward = np.sign(reward)

        # Replay Memoryに遷移を保存
        self.replay_memory.append((state, action, reward, next_state, terminal))
        
        # Replay Memoryが一定数を超えたら、古い遷移から削除
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE_ON_TETRIS:
            # Q Networkの学習
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Target Networkの更新
            if self.t % TARGET_UPDATE_INTERVAL_ON_TETRIS == 0:
                self.sess.run(self.update_target_network)

            # Networkの保存
            if self.t % SAVE_INTERVAL_ON_TETRIS == 0:
                save_path = self.saver.save(self.sess, TRAINED_ADVISER_NETWORK_PATH + '/' + ENV_NAME, global_step=(self.t))
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            # summaryの書き込み
            if self.t >= INITIAL_REPLAY_SIZE_ON_TETRIS:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE_ON_TETRIS:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE_ON_TETRIS <= self.t < INITIAL_REPLAY_SIZE_ON_TETRIS + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1 # タイムステップ

        return next_state


    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Replay Memoryからランダムにミニバッチをサンプル
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # 終了判定をTrueは1に、Falseは0に変換
        terminal_batch = np.array(terminal_batch) + 0
        # Target Networkで次の状態でのQ値を計算
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)}) 
        # 教師信号を計算
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA_ON_TETRIS * np.max(target_q_values_batch, axis=1)

        # 勾配法による誤差最小化
        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss


    def setup_summary(self):
        with tf.variable_scope('Summary'):
            episode_total_reward = tf.Variable(0., name='Total_Reward')
            tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
            episode_avg_max_q = tf.Variable(0., name='Avg_Max_Q')
            tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
            episode_duration = tf.Variable(0., name='Episode')
            tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
            episode_avg_loss = tf.Variable(0., name='Avg_Loss')
            tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
            summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
            summary_placeholders = [tf.placeholder(tf.float32, name='Summary_Placeholder_' + str(i)) for i in range(len(summary_vars))]
            update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
            summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op


    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(TRAINED_ADVISER_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')


    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.t % ACTION_INTERVAL == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            #self.repeated_action = action
            self.repeated_action = 0

        self.t += 1

        return action

