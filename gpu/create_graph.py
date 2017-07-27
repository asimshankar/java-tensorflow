"""Generate a trivial TensorFlow "model".

This script generates the "graph.pb" file in this directory.
Requires TensorFlow for Python. See https://www.tensorflow.org/install/
"""
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[2, 2], name='input')
y = tf.matmul(x, x, name='output')

tf.train.write_graph(tf.get_default_graph(), '.', 'graph.pb', as_text=False)
