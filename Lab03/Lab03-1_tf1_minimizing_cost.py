import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
# 2버전으로 찾아서 수정해보려 하였는데 1버전에 대한 예제 코드만 많이 나와서 1버전 먼저 익힐 것 같다 ㅜ
tf.disable_v2_behavior()

X = [1,2,3]
Y = [1,2,3]

W=tf.placeholder(tf.float32)
# 우리의 가설
hypothesis = X*W


# cost/loss function
# t = [1.,2.,3.,4.]가 주어진 경우 tf.reduce_mean(t) ==> 2.5이다. (평균)
cost = tf.reduce_mean(tf.square(hypothesis-Y))


# 2,3 Run/update graph and get results
# Launch the graph in a session
sess = tf.Session()
# initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# graph로 보여주기 위해 배열에 담아두는 것 같다.
W_val = []
cost_val = []


#Fit the line
for i in range(-30,50) :
    # 0.1 간격으로 움직이겠다
    feed_W = i*0.1
    curr_cost, curr_W = sess.run([cost,W], feed_dict={W : feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# show the graph
plt.plot(W_val, cost_val)
plt.show()