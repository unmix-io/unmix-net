import numpy as np
import math

def normalize(real, imag):
    'Transforms real and imaginary parts to magnitude and angle; scaling it between -1 and 1'

    #real = complex_arr[:, :, 0]
    #imag = complex_arr[:, :, 1]
    cplx = real + imag * 1j

    magnitude = np.abs(cplx)
    angle = np.angle(cplx)

    angle = angle / math.pi # angle is always between -pi and +pi - scale from -1 to 1

    # Todo: Test if log scale performs better
    # magnitude = np.log1p(magnitude)
    max_magnitude = np.max(np.abs(magnitude))
    magnitude = magnitude / (max_magnitude / 2) - 1 # scale from -1 to 1

    return magnitude, angle