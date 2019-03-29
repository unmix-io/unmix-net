#!/usr/bin/env python
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import numpy as np
from PIL import Image
from matplotlib.cm import get_cmap
from keras.callbacks import Callback


class ErrorVisualization(Callback):
    def __init__(self, bot):
        self.bot = bot

    def on_epoch_end(self, epoch, logs={}):
        x_valid = self.bot.x_valid
        y_valid = self.bot.y_valid

        error = np.zeros(y_valid[0].shape)

        n = len(x_valid) // 100
        for i in range(n):
            y_pred = self.bot.model.predict(x_valid[i*100:(i+1)*100],
                                            batch_size=8)
            error += np.sum(np.square(y_pred - y_valid[i*100:(i+1)*100]),
                            axis=0)

        error /= (100*n)
        print(np.mean(error))
        print(np.max(error))
        all_error = error
        if self.bot.config.learn_phase:
            parts = ["real", "imag"]
        else:
            parts = ["amplitude"]
        for i, part in enumerate(parts):
            error = all_error[:, :, i]
            top_val = np.max(error)
            # scale to range 0, 1
            error /= top_val

            cm_hot = get_cmap('plasma')
            im = cm_hot(error)

            # scale to range 0, 255
            im = np.uint8(im * 255)

            im = Image.fromarray(im)
            image_path = os.path.join(self.bot.config.logs, "images")
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            im.save("%s/error%03d-%s-%f.png" %
                    (image_path, epoch, part, top_val), format='PNG')
