# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS


def sparse_update(grads_and_vars,local_grads_and_vars,compression_rate=0.999):
    """Compress gradient to a certain rate"""
    # def quick_topk(grads,)
    def partition(seq):
        pi, seq = seq[0], seq[1:]
        lo = [x for x in seq if x <= pi]
        hi = [x for x in seq if x > pi]
        return lo, pi, hi

    def select(seq, k):
        lo, pi, hi = partition(seq)
        m = len(lo)
        if m == k: return pi
        if m < k: return select(hi, k - m - 1)
        return select(lo, k)

    with tf.name_scope('deep_compression'):
        gradients, variables = zip(*grads_and_vars)
        local_residuals,local_variables = zip(*local_grads_and_vars)
        gradients = list(gradients)
        for i in range(0,len(gradients)):
            gradients[i] = tf.add(gradients[i],local_residuals[i])
        deep_gradients = []
        deep_variables = []
        # sample_rate = (1-comp_rate)*10

        sample_size = math.floor(0.001 * len(gradients) * gradients[0].shape.num_elements())
        sample_grads = tf.reshape(gradients[0],[-1])
        for i in range(1,sample_size):
            sample_grads = tf.concat([sample_grads,tf.reshape(gradients[i],[-1])],0)
            #sample_grads.append(tf.reshape(gradients[i],[-1]))
        temp_threshold = select(sample_grads, math.floor(compression_rate * sample_size))
        for i in range(0,len(gradients)):
            shape = gradients[i].shape
            temp_grads = tf.reshape(gradients[i],[-1])
            #temp_residuals = tf.reshape(local_residuals[i],[-1])
            temp_vars = tf.reshape(variables[i],[-1])
            convert_gradient_array = []
            for j in range(0,temp_grads.shape.num_elements()):
                if(temp_grads[i]>temp_threshold):
                    deep_gradients.append(temp_grads[i])
                    deep_variables.append(temp_vars[i])
                    convert_gradient_array.append(0)
                else:
                    convert_gradient_array.append(temp_grads[i])
            local_residuals[i]= tf.reshape(tf.convert_to_tensor(convert_gradient_array),shape)
        return list(zip(deep_gradients,deep_variables)),list(zip(local_residuals,local_variables))

def init_local_residual(grads_and_vars):
    with tf.name_scope('deep_compression'):
        gradients, variables = zip(*grads_and_vars)
        zero_residuals = []
        for grad in gradients:
            if grad is None:
                zero_residuals.append(None)
            shape = grad.get_shape()
            element = tf.Variable(tf.zeros(shape))
            zero_residuals.append(element)
        return list(zip(zero_residuals,variables))
