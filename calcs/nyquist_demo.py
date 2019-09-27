import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

# sample of 6 Hz continuous
sample_length = 2
sample_frequency = 10
t = np.linspace(0, sample_length, 1000)
y = np.sin(2*np.pi*6*t)

# plt.subplot(2, 1, 2)
plt.plot(t, y, '#2F528F')

n = np.linspace(0, sample_length, sample_length*sample_frequency + 1)
Y = np.sin(2*np.pi*1*n)
markerline, stemlines, baseline = plt.stem(n, Y, basefmt=' ', )
plt.setp(stemlines, 'color', '#FFC000')
plt.setp(markerline, 'color', '#FFC000')

y1 = np.sin(2*np.pi*1*t)
plt.plot(t, y1, '--', color = '#ED7D31')

fig = plt.gcf()
fig.set_size_inches(8, 4)

axes = plt.gca()
axes.axhline(y=0, color='k', linewidth=0.7)
axes.set_facecolor('white')
axes.spines['top'].set_color('white')
axes.spines['right'].set_color('white')

plt.title('Aliasing from incorrect sampling')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Continuous signal', 'Alias signal', 'Discrete sample'], bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0, frameon=False)

axes = plt.gca()
legend = axes.get_legend()
legend.legendHandles[1].set_color('#ED7D31')
legend.legendHandles[2].set_color('#FFC000')
# legend.legendHandles[2].set_colou('#ED7D31')

plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
plt.savefig('nyquist_demo.png')
