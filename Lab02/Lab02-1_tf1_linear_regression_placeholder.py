import tensorflow.compat.v1 as tf
# 2버전으로 찾아서 수정해보려 하였는데 1버전에 대한 예제 코드만 많이 나와서 1버전 먼저 익힐 것 같다 ㅜ
tf.disable_v2_behavior()

X = tf.placeholder(tf.float32,shape=[None])
Y = tf.placeholder(tf.float32,shape=[None])

W = tf.Variable(tf.random.normal([1]),name='weight')
b = tf.Variable(tf.random.normal([1]),name = 'bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


sess = tf.Session()
# initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(2001) :
    cost_val, W_val, b_val, _ = sess.run([cost,W,b,train],
                                         feed_dict={X:[1,2,3,4,5],Y:[2.1,3.1,4.1,5.1,6.1]})
    if step%20 == 0:
        print(step, cost_val,W_val,b_val)


# Testing
print(sess.run(hypothesis,feed_dict={X:[1.4,3.5]}))