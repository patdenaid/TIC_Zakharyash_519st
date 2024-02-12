import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt

# Параметры сигнала
n = 500
Fs = 1000
F_max = 15

# Генерация случайного сигнала
random = np.random.normal(0, 10, n)

# Отсчеты времени
time = np.arange(n)/Fs

# Расчет параметров ФНЧ
w = F_max/(Fs/2)
sos = signal.butter(3, w, 'low', output='sos')

# Фильтрация сигнала
filtered = signal.sosfiltfilt(sos, random)

# Переменные для графиков
fig, ax = plt.subplots()

# Построение графика сигнала
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(time, filtered, linewidth=1)
ax.set_xlabel('Час(секунди)', fontsize=14)
ax.set_ylabel('Амплітуда сигналу', fontsize=14)
plt.title('Сигнал з максимальною частотою F_max=15Гц', fontsize=14)

# Сохранение графика сигнала
fig.savefig('./figures/signal.png', dpi=600)

# Расчет спектра
spectrum = fft.fft(filtered)
spectrum = np.abs(fft.fftshift(spectrum))
freqs = fft.fftfreq(n, 1/Fs)
freqs = fft.fftshift(freqs)

# Построение графика спектра
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(freqs, spectrum, linewidth=1)
ax.set_xlabel('Частота(Гц)', fontsize=14)
ax.set_ylabel('Амплітуда спектру', fontsize=14)
plt.title('Спектр сигналу з максимальною частотою F_max=15Гц', fontsize=14)

# Сохранение графика спектраgit
fig.savefig('./figures/spectrum.png', dpi=600)