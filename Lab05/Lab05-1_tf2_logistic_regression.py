# Lab 5 Logistic Regression Classifier
import tensorflow as tf

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# model 생성
tf.model = tf.keras.Sequential()
# input shape, output shape 정의
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
# use sigmoid activation for 0~1 problem -> activation function 정의
tf.model.add(tf.keras.layers.Activation('sigmoid'))

''' 
better result with loss function == 'binary_crossentropy', try 'mse' for yourself
adding accuracy metric to get accuracy report during training
loss function 정의, optimizer 은 gradient descent로 하고, 평가 척도로 accuracy를 이용 
'''
tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=5000)

# Accuracy report
# history에서 마지막 accuracy 가져온 듯
print("Accuracy: ", history.history['accuracy'][-1])