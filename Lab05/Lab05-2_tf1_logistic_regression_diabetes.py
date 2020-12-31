# Lab 5 Logistic Regression Classifier
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# nomkl을 설치하고 나면 사용할 수 있다!
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# 데이터 분류
xy = np.loadtxt('/Users/yejoonko/git/Computer_Study/AI/tensorflow_basic/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# x,y에 대한 값 분류
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# placeholders for a tensor that will be always fed.
# x가 몇개인지에 따라 8자리의 숫자가 들어감
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 입력 8 개 출력 1개
W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)


    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 0.82794
200 0.755181
400 0.726355
600 0.705179
800 0.686631
...
9600 0.492056
9800 0.491396
10000 0.490767
...
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  0.762846
'''