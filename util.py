# ##############################################################################
#                                                                              #
#  NEATCaptcha - util.py                                                       #
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

from os import mkdir
from os.path import isdir

import numpy as np

from gen import CAPTCHA_HEIGHT, CAPTCHA_LEN, CAPTCHA_LIST, CAPTCHA_WIDTH, gen_captcha_text_and_image


def convert2gray(img):
    if len(img.shape) > 2:
        img = np.mean(img, -1)
    return img


def text2vec(text, captcha_len = CAPTCHA_LEN, captcha_list = CAPTCHA_LIST):
    text_len = len(text)
    if text_len > captcha_len:
        raise ValueError(f'Text length ({text_len}) exceeds captcha length ({captcha_len})')
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(text_len):
        vector[captcha_list.index(text[i]) + i * len(captcha_list)] = 1
    return vector


def vec2text(vec, captcha_list = CAPTCHA_LIST):
    text_list = [captcha_list[int(v)] for v in vec]
    return ''.join(text_list)


def wrap_gen_captcha_text_and_image(shape = (60, 160, 3)):
    while True:
        text, image = gen_captcha_text_and_image()
        if image.shape == shape:
            return text, image


def get_next_batch(batch_count = 60, width = CAPTCHA_WIDTH, height = CAPTCHA_HEIGHT):
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_count):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


if __name__ == '__main__':
    print(get_next_batch(batch_count = 1))
