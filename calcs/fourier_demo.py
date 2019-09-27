import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
plt.rcParams["font.family"] = "serif"

sample_freq = 1000
sample_period = 1/sample_freq

t = np.arange(0, 2*np.pi, sample_period)
y1 = 0.5*np.sin(2*np.pi*1*t)
y10 = np.sin(2*np.pi*10*t)
plt.subplot(2, 1, 1)
plt.plot(t, y1 + y10, color='#2F528F', linewidth = 0.7)
plt.plot(t, y1, '--', color='#FFC000', linewidth = 0.7)
plt.plot(t, y10, '-.', color='#ED7D31', linewidth = 0.7)
fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.title('Time series with comprising pure waveforms')
plt.legend(['Sum', '1 Hz', '10 Hz'], bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0, frameon=False)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
axes = plt.gca()
axes.set_xlim([0, np.pi])
axes.set_facecolor('white')
axes.spines['top'].set_color('white')
axes.spines['right'].set_color('white')

Y = np.fft.fft(y1+y10)
mag = 2*np.abs(Y)/len(t)
freq = np.linspace(0, len(t), len(t))/(2*np.pi)

plt.subplot(2, 1, 2)
plt.plot(freq, mag, '#2F528F', linewidth = 0.7)
fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.title('Corresponding Fourier spectrum')
# plt.legend(['Transformed Sum'], bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0, frameon=False)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
axes = plt.gca()
axes.set_xlim([0, 4*np.pi])
axes.set_ylim([0, 1])
axes.set_facecolor('white')
axes.spines['top'].set_color('white')
axes.spines['right'].set_color('white')

plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)

plt.savefig('fourier_demo.png')
