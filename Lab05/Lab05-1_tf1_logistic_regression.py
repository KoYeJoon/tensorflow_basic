# Lab 5 Logistic Regression Classifier
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)  # for reproducibility

# train_data (multivariable data)
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

# binary (0 or 1)
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders for a tensor that will be always fed.
# X는 (n개, 2개) nx2
# Y는 (n개 , 1개)  nx1
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# w는 X,Y에 의해 2x1 들어오는 것 : 2, 나가는 것 : 1
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
# H(x) = 1/(1+e^(-WT X))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost function 그대로 쓰면 되ㅣㅁ
# cost(W) = ylog(H(x)) + (1-y)(log(1-H(x))의 평균
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# descent algorithm 사용
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
# True = 1.0, false = 0.0으로 해준다.
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            # step, cost함수
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})

    # 학습이 끝난 후
    # hypothesis : 내 모델로 예측한 값
    # correct : 0,1로 분류한 것
    # 실제 Y와 비교
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496
Hypothesis:  [[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  1.0
'''