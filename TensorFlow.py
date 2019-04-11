# https://pytorch.org/tutorials/beginner/examples_autograd/tf_two_layer_net.html#sphx-glr-beginner-examples-autograd-tf-two-layer-net-py

import tensorflow as tf
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
x = tf.placeholder(tf.float32, shape = (None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for the weights and initialize them with random data.
# A TensorFlow Variable persists its value across executions of the graph.
