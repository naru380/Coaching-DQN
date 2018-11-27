import tensorflow as tf
import keras
import numpy as np


class Model():
	def __init__(self):
		self.model, self.a, self.advice = self.build()
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
	
	def build(self):
		model = keras.Sequential()
		model.add(keras.layers.Dense(units=1, input_dim=4, activation='sigmoid', use_bias=False, kernel_initializer='normal'))
		a = tf.placeholder(tf.float32, shape=(1, 4))
		advice = model(a)
		model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
		
		return model, a, advice


	def print_advice(self, action):
		print("advice = {}".format(self.advice.eval(feed_dict={self.a: np.array(action)}, session=self.sess)))

	def learn(self, action, advice):
		for _ in range(100):
			self.model.fit(np.array(action), advice)


def main():
	model = Model()

	action = [[1,0,0,0]]
	advice = [[0.3]]

	model.print_advice(action)
	model.learn(action, advice)
	model.print_advice(action)

if __name__ == '__main__':
	main()
