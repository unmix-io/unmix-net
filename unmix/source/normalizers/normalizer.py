import numpy as np
import math

def normalize(realimag):
    'Transforms real and imaginary parts to magnitude and angle; scaling it between -1 and 1'

    real = realimag[:, :, 0]
    imag = realimag[:, :, 1]
    cplx = real + imag * 1j

    magnitude = np.abs(cplx)
    angle = np.angle(cplx)

    angle = angle / math.pi # angle is always between -pi and +pi - scale from -1 to 1

    # Todo: Test if log scale performs better
    # magnitude = np.log1p(magnitude)
    max_magnitude = np.max(np.abs(magnitude))

    if max_magnitude != 0:
        magnitude = magnitude / (max_magnitude / 2) - 1 # scale from -1 to 1

    combined = np.empty(realimag.shape, dtype=realimag.dtype)
    combined[:, :, 0] = magnitude
    combined[:, :, 1] = angle
    return combined