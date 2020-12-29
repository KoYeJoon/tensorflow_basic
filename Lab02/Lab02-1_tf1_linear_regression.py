#import tensorflow as tf
import tensorflow.compat.v1 as tf
# 2버전으로 찾아서 수정해보려 하였는데 1버전에 대한 예제 코드만 많이 나와서 1버전 먼저 익힐 것 같다 ㅜ
tf.disable_v2_behavior()


# 1. build graph using TF operations
# H(x) = Wx + b
# X and Y data (1,1), (2,2), (3,3)
x_train = [1,2,3]
y_train = [1,2,3]


# 최초 값은 모르므로 값이 하나인 랜덤 variable을 설정해주기 위함
W = tf.Variable(tf.random.normal([1]),name='weight')
b = tf.Variable(tf.random.normal([1]),name = 'bias')


# 우리의 가설
hypothesis = x_train * W + b


# cost/loss function
# t = [1.,2.,3.,4.]가 주어진 경우 tf.reduce_mean(t) ==> 2.5이다. (평균)
cost = tf.reduce_mean(tf.square(hypothesis-y_train))


# cost minimize (train-cost-hypothesis-W,b 연결됨)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)



# 2,3 Run/update graph and get results
# Launch the graph in a session
sess = tf.Session()
# initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(2001) :
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))
