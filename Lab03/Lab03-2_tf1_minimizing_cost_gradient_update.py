# Lab 3 Minimizing Cost
import tensorflow.compat.v1 as tf
# 2버전으로 찾아서 수정해보려 하였는데 1버전에 대한 예제 코드만 많이 나와서 1버전 먼저 익힐 것 같다 ㅜ
tf.disable_v2_behavior()

tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data
# We know that W should be 1
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name="weight")

# 나중에 값을 넘겨주고자 함 .
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# sgd 식과 같으므로 하나씩 차근차근 식을 수행하는 느낌..!
# 이는 optimizer를 통해 cost를 미분하지 않아도 자동으로 어낼 수 있다..! ==> 03-3 참고
# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
# 바로 할당이 안되므로 아래와 같은 방법으로 W 값을 update한다.
update = W.assign(descent)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(21):
        _, cost_val, W_val = sess.run(
            [update, cost, W], feed_dict={X: x_data, Y: y_data}
        )
        print(step, cost_val, W_val)