# Lab 4 Multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)  # for reproducibility


# 1. File 들의 list 생성 , file 여러 개 가능, shuffle 여부도 설정 가능
filename_queue = tf.train.string_input_producer(
    ['/Users/yejoonko/git/Computer_Study/AI/tensorflow_basic/data-01-test-score.csv'], shuffle=False, name='filename_queue')

# 2. 읽어올 reader 정의
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


# 3. 읽어온 value를 어떻게 읽어오고 parsing할 지 decoder정의
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# 각 field의 data type을 정의해준다.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# tf.train.batch -> batch로 읽어옴
# collect batches of csv in
# indexing
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# X,Y,W shape은 신경써주어야 한다.
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    # 펌프질하여 데이터 가져옴
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[185.33531]]
Other scores will be  [[178.36246]
 [177.03687]]
'''