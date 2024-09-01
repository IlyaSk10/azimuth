import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft

import math


import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

phase = np.pi / 4
t = np.linspace(0, 10, num=200, endpoint=False)
y = np.cos(2 * np.pi * t + phase)
Y = scipy.fftpack.fftshift(scipy.fftpack.fft(y))
f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(t)))

p = np.angle(Y)
p[np.abs(Y) < 1] = 0
plt.plot(f, p)
plt.show()

fd = 500
f = 5
tmod = 2

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
# phase = np.arctan(np.imag(fft_sc) / np.real(fft_sc)) * 180 / np.pi
# phase = phase[:phase.shape[0] // 2]

f = [(i / len(t)) * fd for i in range(int(len(t) / 2))]
plt.plot(np.angle(fft_sc, deg=True))
fft_sc=fft_sc[:500]
#plt.plot(f, np.arctan(-np.imag(fft_sc) / np.real(fft_sc)) * 180 / np.pi)
#plt.plot(f, 2*((np.angle(fft_sc[:500]) * 180) / np.pi))
plt.show()
pass
