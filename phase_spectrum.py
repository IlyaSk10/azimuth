import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# https://stackoverflow.com/questions/54454723/scipy-fft-how-to-get-phase-angle

fd = 5000
f = 5
tmod = 2
shift = 0
phase = np.pi / 2

t = np.linspace(0, tmod, fd * tmod)
y = np.cos(2 * np.pi * f * t + shift + phase)

# t = np.linspace(0, 10, num=200, endpoint=False)
# y = np.sin(2 * np.pi * t + phase)

# Y = scipy.fftpack.fftshift(scipy.fftpack.fft(y))
Y = scipy.fftpack.fft(y)
# f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(t)))
fr = [(i / len(t)) * fd for i in range(int(len(t) / 2))]

#p = np.arctan(np.imag(Y / np.real(Y)))
p = np.angle(Y)
p[np.abs(Y) < 100] = 0
p = p[:int(len(t) / 2)]

plt.plot(fr, p * 180 / np.pi)
plt.show()

fd = 500
f = 5
tmod = 2
phase = np.pi / 2

t = np.linspace(0, tmod, fd * tmod)
y = np.sin(2 * np.pi * f * t + phase)

fft_sc = scipy.fftpack.fft(y)

fr = [(i / len(t)) * fd for i in range(int(len(t) / 2))]
p = np.arctan(np.imag(fft_sc) / np.real(fft_sc))
p[np.abs(fft_sc) < 10] = 0
p = p[:int(len(t) / 2)]
plt.plot(fr, -p * 180 / np.pi)
plt.show()

pass
