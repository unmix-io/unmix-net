import numpy as np
import math

def normalize(real, imag):

    #real = complex_arr[:, :, 0]
    #imag = complex_arr[:, :, 1]
    cplx = real + imag * 1j

    magnitude = np.abs(cplx)
    angle = np.angle(cplx)

    angle = angle / math.pi # angle is always between -pi and +pi - scale from -1 to 1

    # Todo: Test if log scale performs better
    # magnitude = np.log1p(magnitude)
    magnitude = magnitude / np.max(np.abs(magnitude)) # scale from -1 to 1

    return magnitude, angle