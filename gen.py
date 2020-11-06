# ##############################################################################
#                                                                              #
#  NEATCaptcha - gen.py                                                        #
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

import random
from os import mkdir
from os.path import isdir

import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha

NUMBERS = '0123456789'
LOW_CASE = 'abcdefghijklmnopqrstuvwxyz'
UP_CASE = LOW_CASE.upper()
ALL = NUMBERS + LOW_CASE + UP_CASE

CAPTCHA_LIST = ALL
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160


def random_captcha_text(char_set = CAPTCHA_LIST, captcha_size = CAPTCHA_LEN):
    return ''.join([random.choice(char_set) for _ in range(captcha_size)])


def gen_captcha_text_and_image(width = CAPTCHA_WIDTH, height = CAPTCHA_HEIGHT, save = None):
    image = ImageCaptcha(width = width, height = height)
    captcha_text = random_captcha_text()
    captcha = image.generate(captcha_text)
    if save:
        if not isdir('./captcha'):
            mkdir('./captcha')
        image.write(captcha_text, './captcha/' + captcha_text + '.jpg')
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    for i in range(10):
        text, image = gen_captcha_text_and_image(save = True)
        print(text, image.shape)
