import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft

import math

fd = 1000
f = 24
tmod = 4

t = np.linspace(0, tmod, fd * tmod)
y = np.sin(2 * np.pi * f * t + np.pi/4)

# plt.plot(t, y)
# plt.show()

spectrum = np.abs(np.fft.fft(y))
spectrum[0] = spectrum[0] / (fd * tmod)
spectrum = spectrum[:spectrum.shape[0] // 2]
spectrum[1:] = 2 * spectrum[1:] / (fd * tmod)

# fft = np.fft.fft(y)
fft_sc = fft(y)
phase = np.arctan(np.imag(fft_sc) / np.real(fft_sc)) * 180 / np.pi
# phase = phase[:phase.shape[0] // 2]

f = [(i / len(t)) * fd for i in range(int(len(t) / 2))]
plt.plot(phase, marker='*')
plt.show()
pass
