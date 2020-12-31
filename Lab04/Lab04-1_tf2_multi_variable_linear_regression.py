# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

tf.model = tf.keras.Sequential()

# 객체를 생성하면서 output, input shape 정의
# unit : 출력층의 shape, input_dim : 입력층의 shape
# activationn function 정의
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))  # input_dim=3 gives multi-variable regression
tf.model.add(tf.keras.layers.Activation('linear'))  # this line can be omitted, as linear activation is default
# advanced reading https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

# loss function 정의
# 랜덤하게 추출하여 일부 데이터에 대해 가중치 조절하는 optimizer 정의하여 객체 전달
# optimizer : optimizer 객체 전달, loss : 손실함수 커스텀
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
tf.model.summary()

# training
history = tf.model.fit(x_data, y_data, epochs=100)

# test
y_predict = tf.model.predict(np.array([[72., 93., 90.]]))
print(y_predict)