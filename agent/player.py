# coding: utf-8

from .common import *

class Player():
	def __init__(self, num_actions):
		self.num_actions = num_actions # 行動数
		self.epsilon = INITIAL_EPSILON # ε-greedy法のεの初期化
		self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS # εの減少率
		self.t = 0 # タイムステップ
		self.repeated_action = 0 # フレームスキップ間にリピートする行動を保持するための変数
		#self.average_total_reward = 0
		#self.advise_memory = np.array([[]])
		#self.action_memory = np.array([[]])
		
		# Replay Memoryの構築
		self.replay_memory = deque()

		# summaryに使用するパラメータ
		self.total_reward = 0
		self.total_q_max = 0
		self.total_loss = 0
		self.duration = 0
		self.episode = 0

		# クラス専用のグラフを構築する
		self.graph = tf.Graph()
		
		with self.graph.as_default():
			with tf.variable_scope("Player"):
				with tf.variable_scope("Q_Network"):
					# Q Networkの構築
					self.s, self.ad, self.q_values, q_network = self.build_network()
					q_network_weights = q_network.trainable_weights

				with tf.variable_scope("Target_Network"):
					# Target Networkの構築
					self.st, self.adt, self.target_q_values, target_network = self.build_network()
					target_network_weights = target_network.trainable_weights

				# 定期的にTarget Networkを更新するための処理の構築
				self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

				# 誤差関数や最適化のための処理の構築
				self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

				"""
				# Language Networkの構築
				with tf.variable_scope("Langage_Network"):
					self.la, self.action, self.language_network = self.build_language_network()
				"""

			# Sessionの構築
			#self.sess = tf.InteractiveSession()
			self.sess = tf.Session(graph=self.graph)
			
			self.saver = tf.train.Saver(q_network_weights)
			self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
			self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

			if not os.path.exists(SAVE_NETWORK_PATH):
				os.makedirs(SAVE_NETWORK_PATH)

			# 変数の初期化(Q Networkの初期化)
			self.sess.run(tf.global_variables_initializer())
			#sess.run(tf.global_variables_initializer())

			# Target Networkの初期化
			self.sess.run(self.update_target_network)

		# debug
		summary_writer = tf.summary.FileWriter('data', graph=self.sess.graph)


	def build_network(self):
		# ~/.keras/keras.jsonのimage_data_formatを'channel_last'から'channel_first'に変更
		
		display_input = Input(shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=tf.float32)
		advice_input = Input(shape=(1, ))

		x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(display_input)
		x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
		x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
		x = Flatten()(x)

		merged = concatenate([x, advice_input])

		y = Dense(512, activation='relu')(merged)
		y = Dense(self.num_actions)(y)

		model = Model(inputs=[display_input, advice_input], outputs=y)

		s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
		ad = tf.placeholder(tf.float32, (None, 1))
		q_values = model(inputs=[s, ad])

		"""
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.num_actions))

		s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
		q_values = model(s)
		"""

		return s, ad, q_values, model


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


	def get_action(self, state, advice):
		action = self.repeated_action # 行動をリピート

		if self.t % ACTION_INTERVAL == 0:
			if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
				action = random.randrange(self.num_actions) # ランダムに行動を選択
			else:
				action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)], self.ad:advice}, session=self.sess))
			self.repeated_action = action # フレームスキップ間にリピートする行動を格納

		# εを線形に減少させる
		if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
			self.epsilon -= self.epsilon_step

		return action


	def run(self, state, action, advice, reward, terminal, observation):
		# 次の状態を作成
		next_state = np.append(state[1:, :, :], observation, axis=0)
		#print(next_state)

		# 報酬を固定、正は1、負は−1、0はそのまま
		reward = np.sign(reward)

		# Replay Memoryに遷移を保存
		self.replay_memory.append((state, action, [float(advice)], reward, next_state, terminal))
		
		# Replay Memoryが一定数を超えたら、古い遷移から削除
		if len(self.replay_memory) > NUM_REPLAY_MEMORY:
			self.replay_memory.popleft()

		if self.t >= INITIAL_REPLAY_SIZE:
			# Q Networkの学習
			if self.t % TRAIN_INTERVAL == 0:
				self.train_network()

			# Target Networkの更新
			if self.t % TARGET_UPDATE_INTERVAL == 0:
				self.sess.run(self.update_target_network)

			# Networkの保存
			if self.t % SAVE_INTERVAL == 0:
				save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=(self.t))
				print('Successfully saved: ' + save_path)

		self.total_reward += reward
		self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)], self.ad: advice}, session=self.sess))
		self.duration += 1

		if terminal:
			# summaryの書き込み
			if self.t >= INITIAL_REPLAY_SIZE:
				stats = [self.total_reward, self.total_q_max / float(self.duration),
						self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
				for i in range(len(stats)):
					self.sess.run(self.update_ops[i], feed_dict={
						self.summary_placeholders[i]: float(stats[i])
					})
				summary_str = self.sess.run(self.summary_op)
				self.summary_writer.add_summary(summary_str, self.episode + 1)

			# Debug
			if self.t < INITIAL_REPLAY_SIZE:
				mode = 'random'
			elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
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

		"""
		advised_action = self.action.eval(feed_dict={self.la: advise}, session=self.sess)
		self.advised_action = advised_action
		rand = (1 - (-1)) * np.random.rand(advised_action.size).reshape(advised_action.shape) + (-1)
		teacher_signal = advised_action + rand
		teacher_signal[teacher_signal >= 1] = 0.9999999
		teacher_signal[teacher_signal < 0] = 0
		
		self.action_memory = np.append(self.action_memory, teacher_signal)
		self.advise_memory = np.append(self.advise_memory, advise)

		if terminal == True:
			self.total_reward += reward
		else:
			if self.total_reward <= self.average_total_reward:
				# オンライン学習
				#self.language_network.fit(np.array(action), teacher_signal, verbose=1)
				# バッチ学習
				self.train_language_network()
			self.average_total_reward = (self.average_total_reward * (self.episode - 1) + self.total_reward) / self.episode
			self.total_reward = 0
			self.episode += 1
			self.advise_memory = np.array([[]])
			self.action_memory = np.array([[]])
		"""

		return next_state


	def train_network(self):
		state_batch = []
		action_batch = []
		advice_batch = []
		reward_batch = []
		next_state_batch = []
		terminal_batch = []
		y_batch = []

		# Replay Memoryからランダムにミニバッチをサンプル
		minibatch = random.sample(self.replay_memory, BATCH_SIZE)
		for data in minibatch:
			state_batch.append(data[0])
			action_batch.append(data[1])
			advice_batch.append(data[2])
			reward_batch.append(data[3])
			next_state_batch.append(data[4])
			terminal_batch.append(data[5])

		# 終了判定をTrueは1に、Falseは0に変換
		terminal_batch = np.array(terminal_batch) + 0
		# Target Networkで次の状態でのQ値を計算
		target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0), self.adt: np.float32(np.array(advice_batch))}, session=self.sess) 
		# 教師信号を計算
		y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

		# 勾配法による誤差最小化
		loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
			self.s: np.float32(np.array(state_batch) / 255.0),
			self.ad: np.float32(np.array(advice_batch)),
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


	"""
	def build_language_network(self):
		model = Sequential()
		#model.add(Dense(units=5, activation='sigmoid', input_dim=1, init='normal'))
		model.add(Dense(units=self.num_actions, activation='softmax', kernel_initializer='normal'))
		#model.add(Dense(units=1, activation='softmax'))
		#a = tf.placeholder(tf.float32, shape=(self.num_actions, 1), name='advise')
		a = tf.placeholder(tf.float32, shape=(None, 1), name='advise')
		action = model(a)

		model.compile(loss='mean_squared_error',  optimizer=SGD(lr=0.1), metrics=['accuracy'])

		return a, action, model
	"""

	"""
	def get_action_from_advise(self, advice):
		action = self.action.eval(feed_dict={self.la: advice}, session=self.sess)

		return action
	"""

	"""
	def train_language_network(self):
		K.set_session(self.sess)
		#print("advised_action after = {}".format(np.argmax(self.advised_action)))
		self.language_network.fit(np.reshape(np.array(self.advise_memory), (-1, 1)), np.reshape(self.action_memory, (-1, self.num_actions)), epochs=1, verbose=1)
    	#print("advised_action before = {}".format(np.argmax(self.advised_action)))
		print("weights after lerning is \n{}".format(self.language_network.get_weights()))
	"""
