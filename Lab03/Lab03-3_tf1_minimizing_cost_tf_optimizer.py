# Lab 3 Minimizing Cost
import tensorflow.compat.v1 as tf
# 2버전으로 찾아서 수정해보려 하였는데 1버전에 대한 예제 코드만 많이 나와서 1버전 먼저 익힐 것 같다 ㅜ
tf.disable_v2_behavior()

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# 일부러 말도안되는 값을 넣어 잘 내려가는지 확인함
# Set wrong model weights
W = tf.Variable(5.0)

# Linear model
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer로 손쉽게 업데이트 가능!
# Minimize: Gradient Descent Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, W_val = sess.run([train, W])
        print(step, W_val)