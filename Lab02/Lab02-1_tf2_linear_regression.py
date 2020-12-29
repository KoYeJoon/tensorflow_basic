import numpy as np
import tensorflow as tf

# train_data 정의
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# keras 에 있는 sequential 함수를 가져온다.
tf.model = tf.keras.Sequential()

# 객체를 생성하면서 output, input shape 정의
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# 랜덤하게 추출하여 일부 데이터에 대해 가중치 조절
sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate

# loss function 정의
# optimizer : optimizer 객체 전달, loss : 손실함수 커스텀
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2

#모델 요약
# prints summary of the model to the terminal
tf.model.summary()

# training 시작
# fit() executes training
tf.model.fit(x_train, y_train, epochs=200)

# 예측
# predict() returns predicted value
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)