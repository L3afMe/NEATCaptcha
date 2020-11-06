# ##############################################################################
#                                                                              #
#  NEATCaptcha - test.py                                                       #
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

from os.path import isdir

from PIL import Image
from tensorflow import Session, argmax, float32, placeholder, reshape
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.compat.v1.train import Saver

from gen import gen_captcha_text_and_image
from train import cnn_graph
from util import CAPTCHA_HEIGHT, CAPTCHA_LEN, CAPTCHA_LIST, CAPTCHA_WIDTH, convert2gray, vec2text


def captcha2text(image_list, height = CAPTCHA_HEIGHT, width = CAPTCHA_WIDTH):
    if not isdir('./model'):
        print('Model directory does not exists.')
        return
    x = placeholder(float32, [None, height * width])
    keep_prob = placeholder(float32)
    y_conv = cnn_graph(x, keep_prob, (height, width))
    saver = Saver()
    with Session() as sess:
        saver.restore(sess, latest_checkpoint('./model/'))
        predict = argmax(reshape(y_conv, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), 2)
        vector_list = sess.run(predict, feed_dict = {x: image_list, keep_prob: 1})
        vector_list = vector_list.tolist()
        text_list = [vec2text(vector) for vector in vector_list]
        return text_list


if __name__ == '__main__':
    text, image = gen_captcha_text_and_image()
    img = Image.fromarray(image)
    image = convert2gray(image)
    image = image.flatten() / 255
    pre_text = captcha2text([image])
    print("Text:", text, ', Actual Text:', pre_text, ', Result:', pre_text[0] == text)
    img.show()
