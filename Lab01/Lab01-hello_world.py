# tf.__version__ '2.2.0'
import tensorflow as tf

#constant 생성 graph 속 node가 Hello, TensorFlow가 생성된 것임 !
hello = tf.constant("Hello, TensorFlow!")

# hello node 실행
tf.print(hello)