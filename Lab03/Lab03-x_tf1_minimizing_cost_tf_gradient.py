# Lab 3 Minimizing Cost
# gradient 를 조정하고 싶은 경우
import tensorflow.compat.v1 as tf
# 2버전으로 찾아서 수정해보려 하였는데 1버전에 대한 예제 코드만 많이 나와서 1버전 먼저 익힐 것 같다 ㅜ
tf.disable_v2_behavior()

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.)

# Linear model
hypothesis = X * W

# Manual gradient
# 미분하면 1/2생기므로 *2 해줌 (상관없으므로 *2 해줘도 됨)
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Optimizer
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# 위 처럼 바로 minimize하지 않고..!
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 1. gradient 계산한 값을 받는다.
# Get gradients
gvs = optimizer.compute_gradients(cost)

# 2. 원하는 대로 수정가능
# Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

# 3. 조정한 값으로 적용
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        gradient_val, gvs_val, _ = sess.run([gradient, gvs, apply_gradients])
        print(step, gradient_val, gvs_val)