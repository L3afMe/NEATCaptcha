# ##############################################################################
#                                                                              #
#  NEATCaptcha - train.py                                                      #
#  Copyright (C) 2020 L3af                                                     #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify it     #
#  under the terms of the GNU General Public License as published by the       #
#  Free Software Foundation, either version 3 of the License, or (at your      #
#  option) any later version.                                                  #
#                                                                              #
#  This program is distributed in the hope that it will be useful, but         #
#  WITHOUT ANY WARRANTY; without even the implied warranty of                  #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See  the GNU General   #
#  Public License for more details.                                            #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program. If not, see <https://www.gnu.org/licenses/>.       #
#                                                                              #
# ##############################################################################

from datetime import datetime
from os import mkdir
from os.path import isdir

from tensorflow import Variable, argmax, cast, equal, float32, matmul, reduce_mean, reshape
from tensorflow.compat.v1 import Session, placeholder, global_variables_initializer
from tensorflow.compat.v1.train import AdamOptimizer, Saver
from tensorflow.nn import conv2d as _conv2d, dropout, max_pool2d as _max_pool2d, relu, sigmoid_cross_entropy_with_logits
from tensorflow.random import normal

from gen import CAPTCHA_HEIGHT, CAPTCHA_LEN, CAPTCHA_LIST, CAPTCHA_WIDTH
from util import get_next_batch


def weight_variable(shape, w_alpha = 0.01):
    return Variable(w_alpha * normal(shape))


def bias_variable(shape, b_alpha = 0.1):
    return Variable(b_alpha * normal(shape))


def conv2d(x, w):
    return _conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')


def max_pool2d(x):
    return _max_pool2d(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


def cnn_graph(x, keep_prob, size, captcha_list = CAPTCHA_LIST, captcha_len = CAPTCHA_LEN):
    x_image = reshape(x, shape = [-1, size[0], size[1], 1])
    w_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool2d(h_conv1)
    h_drop1 = dropout(h_pool1, rate = 1 - keep_prob)
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = relu(conv2d(h_drop1, w_conv2) + b_conv2)
    h_pool2 = max_pool2d(h_conv2)
    h_drop2 = dropout(h_pool2, rate = 1 - keep_prob)
    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = relu(conv2d(h_drop2, w_conv3) + b_conv3)
    h_pool3 = max_pool2d(h_conv3)
    h_drop3 = dropout(h_pool3, rate = 1 - keep_prob)
    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])
    w_fc = weight_variable([image_height * image_width * 64, 1024])
    b_fc = bias_variable([1024])
    h_drop3_re = reshape(h_drop3, [-1, image_height * image_width * 64])
    h_fc = relu(matmul(h_drop3_re, w_fc) + b_fc)
    h_drop_fc = dropout(h_fc, rate = 1 - keep_prob)
    w_out = weight_variable([1024, len(captcha_list) * captcha_len])
    b_out = bias_variable([len(captcha_list) * captcha_len])
    y_conv = matmul(h_drop_fc, w_out) + b_out
    return y_conv


def optimize_graph(y, y_conv):
    return AdamOptimizer(1e-3).minimize(
            reduce_mean(sigmoid_cross_entropy_with_logits(labels = y, logits = y_conv)))


def accuracy_graph(y, y_conv, width = len(CAPTCHA_LIST), height = CAPTCHA_LEN):
    return reduce_mean(cast(equal(argmax(reshape(y_conv, [-1, height, width]), 2),
                                  argmax(reshape(y, [-1, height, width]), 2)), float32))


def train(height = CAPTCHA_HEIGHT, width = CAPTCHA_WIDTH, y_size = len(CAPTCHA_LIST) * CAPTCHA_LEN):
    acc_rate = 0.95
    
    x = placeholder(float32, [None, height * width])
    y = placeholder(float32, [None, y_size])
    keep_prob = placeholder(float32)
    y_conv = cnn_graph(x, keep_prob, (height, width))
    optimizer = optimize_graph(y, y_conv)
    accuracy = accuracy_graph(y, y_conv)
    saver = Saver()
    sess = Session()
    sess.run(global_variables_initializer())
    step = 0
    while 1:
        batch_x, batch_y = get_next_batch(64)
        sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y, keep_prob: 0.75})
        if step % 100 == 0:
            batch_x_test, batch_y_test = get_next_batch(100)
            acc = sess.run(accuracy, feed_dict = {x: batch_x_test, y: batch_y_test, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            if acc > acc_rate:
                if not isdir('./model'):
                    mkdir('./model')
                
                model_path = "./model/captcha.model"
                saver.save(sess, model_path, global_step = step)
                acc_rate += 0.005
                if acc_rate >= 1:
                    break
        step += 1
    sess.close()


if __name__ == '__main__':
    train()
