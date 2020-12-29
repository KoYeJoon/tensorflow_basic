# Lab 3 Minimizing Cost
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# data 정의
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# keras의 모델을 가져옴
tf.model = tf.keras.Sequential()

# 객체를 생성하면서 output, input shape 정의
# unit : 출력층의 shape, input_dim : 입력층의 shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# 랜덤하게 추출하여 일부 데이터에 대해 가중치 조절
sgd = tf.keras.optimizers.SGD(lr=0.1)

# loss function 정의
# optimizer : optimizer 객체 전달, loss : 손실함수 커스텀
tf.model.compile(loss='mse', optimizer=sgd)

# 모델 요약
tf.model.summary()

# 모델 훈련
# fit() trains the model and returns history of train
history = tf.model.fit(x_train, y_train, epochs=100)

# test data에 대한 값 예측
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)

# 그래프 그리기
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
