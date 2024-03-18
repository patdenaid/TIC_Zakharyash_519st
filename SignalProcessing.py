import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
import os

def згенерувати_сигнал(n, максимальна_частота, Fs, F_фільтр):
    випадковий_сигнал = np.random.normal(0, 10, n)
    часові_значення = np.arange(n) / Fs

    # Застосування низькочастотного фільтра
    w_max = максимальна_частота / (Fs / 2)
    low_filter_params = signal.butter(3, w_max, 'low', output='sos')
    low_filtered_signal = signal.sosfiltfilt(low_filter_params, випадковий_сигнал)

    # Застосування високочастотного фільтра
    w_filter = F_фільтр / (Fs / 2)
    high_filter_params = signal.butter(3, w_filter, 'high', output='sos')
    band_filtered_signal = signal.sosfiltfilt(high_filter_params, low_filtered_signal)

    # Обрізка сигналу до потрібної смуги частот
    filtered_signal = band_filtered_signal - low_filtered_signal

    # Додаємо крок дискретизації
    крок_дискретизації = 1 / Fs

    # Відновлюємо сигнал за допомогою інтерполяції
    upsampled_time_values = np.arange(0, n / Fs, крок_дискретизації / 10)
    upsampled_signal = np.interp(upsampled_time_values, часові_значення, filtered_signal)

    return часові_значення, filtered_signal, upsampled_time_values, upsampled_signal, крок_дискретизації

def обчислити_спектр(сигнал, Fs):
    N = len(сигнал)
    fft_сигнал = np.fft.fft(сигнал)
    спектр = 2 * np.abs(fft_сигнал[:N // 2]) / N
    частоти = np.arange(0, Fs / 2, Fs / N)
    return частоти, спектр

# Параметри сигналу
n = 500
Fs = 1000
F_max = 15

# Генерація випадкового сигналу
випадковий = np.random.normal(0, 10, n)

# Відліки часу
час = np.arange(n) / Fs

# Розрахунок параметрів ФНЧ
w = F_max / (Fs / 2)
sos = signal.butter(3, w, 'low', output='sos')

# Фільтрація сигналу
filtered = signal.sosfiltfilt(sos, випадковий)

# Змінні для графіків
fig, ax = plt.subplots()

# Побудова графіка сигналу
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(час, filtered, linewidth=1)
ax.set_xlabel('Час(секунди)', fontsize=14)
ax.set_ylabel('Амплітуда сигналу', fontsize=14)
plt.title('Сигнал з максимальною частотою F_max=15Гц', fontsize=14)

# Збереження графіка сигналу
if not os.path.exists("figures"):
    os.makedirs("figures")
fig.savefig('./figures/signal.png', dpi=600)

# Розрахунок спектра
спектр = fft.fft(filtered)
спектр = np.abs(fft.fftshift(спектр))
частоти = fft.fftfreq(n, 1/Fs)
частоти = fft.fftshift(частоти)

# Побудова графіка спектра
fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(частоти, спектр, linewidth=1)
ax.set_xlabel('Частота(Гц)', fontsize=14)
ax.set_ylabel('Амплітуда спектру', fontsize=14)
plt.title('Спектр сигналу з максимальною частотою F_max=15Гц', fontsize=14)

# Збереження графіка спектра
fig.savefig('./figures/spectrum.png', dpi=600)

# Генерація сигналів з різними рівнями квантування
максимальна_частота = 15 # Максимальна частота сигналу
Fs = 1000  # Частота дискретизації
F_фільтр = 22  # Полоса пропуску фільтра
часові_значення, filtered_signal, upsampled_time_values, upsampled_signal, крок_дискретизації = згенерувати_сигнал(n, максимальна_частота, Fs, F_фільтр)

# Обчислення спектрів
частоти, signal_spectrum = обчислити_спектр(filtered_signal, Fs)
upsampled_frequencies, upsampled_spectrum = обчислити_спектр(upsampled_signal, Fs * 10)

# Відображення сигналів та їх спектрів
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))
ax1.plot(часові_значення, filtered_signal)
ax1.set_title('Вхідний сигнал')
ax1.set_xlabel('Час')
ax1.set_ylabel('Амплітуда')

ax2.plot(upsampled_time_values, upsampled_signal)
ax2.set_title('Відновлений сигнал')
ax2.set_xlabel('Час')
ax2.set_ylabel('Амплітуда')

ax3.plot(частоти, signal_spectrum)
ax3.set_title(f'Спектр вихідного сигналу (Fs = {Fs} Гц)')
ax3.set_xlabel('Частота (Гц)')
ax3.set_ylabel('Амплітуда')

ax4.plot(upsampled_frequencies, upsampled_spectrum)
ax4.set_title(f'Спектр відновленого сигналу (Fs = {Fs * 10} Гц)')
ax4.set_xlabel('Частота (Гц)')
ax4.set_ylabel('Амплітуда')

plt.tight_layout()
plt.savefig("figures/Сигнали та їх спектри.png", dpi=300)
plt.show()

# Визначення M_values
M_values = [4, 16, 64, 256]

# Відображення результатів квантування та розрахунку дисперсій
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
variances = []  # Визначення масиву для збереження дисперсій
snr_ratios = []  # Визначення масиву для збереження співвідношень сигнал-шум

for s in range(4):
    M = M_values[s]
    delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)
    quantize_signal = delta * np.round(filtered_signal / delta)

    # Розрахунок бітових послідовностей
    quantize_levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + 1, delta)
    quantize_bit = [format(bits, '0' + str(int(np.log2(M))) + 'b') for bits in np.arange(0, M)]
    quantize_table = np.c_[quantize_levels[:M], quantize_bit[:M]]

    # Збереження таблиці квантування
    fig_table, ax_table = plt.subplots(figsize=(14 / 2.54, M / 2.54))
    table = ax_table.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'],
                           loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax_table.axis('off')
    plt.savefig(f"./figures/Таблиця квантування для {M} рівнів.png", dpi=600)
    plt.close(fig_table)

    # Кодування сигналу
    bits = []
    for signal_value in quantize_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if np.round(np.abs(signal_value - value), 0) == 0:
                bits.append(quantize_bit[index])
                break

    bits = [int(item) for item in list(''.join(bits))]

    # Побудова графіку бітових послідовностей
    fig_bits, ax_bits = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax_bits.step(np.arange(0, len(bits)), bits, linewidth=0.1)
    ax_bits.set_xlabel('Відліки')
    ax_bits.set_ylabel('Бітова послідовність')
    ax_bits.set_title(f'Бітова послідовність для {M} рівнів квантування')
    plt.savefig(f"./figures/Бітова послідовність для {M} рівнів.png", dpi=600)
    plt.close(fig_bits)

    # Збереження даних для графіків
    variances.append(np.var(quantize_signal))
    snr_ratios.append(np.var(filtered_signal) / np.var(quantize_signal))

    # Відображення цифрового сигналу
    i, j = divmod(s, 2)
    ax[i][j].step(часові_значення, quantize_signal, linewidth=1, where='post', label=f'M = {M}')

fig.suptitle("Цифрові сигнали з різними рівнями квантування", fontsize=14)
fig.supxlabel("Час", fontsize=14)
fig.supylabel("Амплітуда цифрового сигналу", fontsize=14)

# Збереження зображення
plt.savefig("figures/Цифрові сигнали з різними рівнями квантування.png", dpi=600)

# Показати графики
plt.show()

# Графік дисперсії
fig_variance, ax_variance = plt.subplots(figsize=(10, 6))
ax_variance.plot(M_values, variances, marker='o', color='b', label='Дисперсія цифрового сигналу')
ax_variance.set_xlabel('Кількість рівнів квантування')
ax_variance.set_ylabel('Дисперсія')
ax_variance.set_xscale('log', base=2)
ax_variance.legend()
plt.title("Залежність дисперсії цифрового сигналу від кількості рівнів квантування")

# Збереження графіку
plt.savefig("figures/Дисперсія цифрового сигналу.png", dpi=600)

# Показати графік
plt.show()

# Графік співвідношення сигнал-шум
fig_snr, ax_snr = plt.subplots(figsize=(10, 6))
ax_snr.plot(M_values, snr_ratios, marker='o', color='r', label='Співвідношення сигнал-шум')
ax_snr.set_xlabel('Кількість рівнів квантування')
ax_snr.set_ylabel('Співвідношення сигнал-шум')
ax_snr.set_xscale('log', base=2)
ax_snr.legend()
plt.title("Залежність співвідношення сигнал-шум від кількості рівнів квантування")

# Збереження графіку
plt.savefig("figures/Співвідношення сигнал-шум.png", dpi=600)

# Показати графік
plt.show()
