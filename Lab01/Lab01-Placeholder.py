import tensorflow as tf
#2 이상 버전부터 placeholder -> Variable
#feed_dict 없어짐
#sess.run 사라짐->함수를 통해 실행
W = tf.Variable(tf.ones(shape=(2,2)),name='W')
b = tf.Variable(tf.zeros(shape=(2)),name='b')


@tf.function
def forward(x):
    return W*x+b

out_a = forward([1,0])
print(out_a)

# a = tf.placeholder(tf.float32)
# b= tf.placeholder(tf.float32)
# adder_node = a+b

# print(sess.run(adder_node, feed_dict={a:3,b:4.5}))
# print(sess.run(adder_node, feed_dict = {a:[1,3], b:[2,4]}))

