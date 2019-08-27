import numpy as np
import matplotlib.pyplot as plt

def as_cpp_var(fsum, cufft=0):

    output_file = open('time_data_cpp.txt', 'w')
    output_file.write('float ') if not cufft else output_file.write('cufftReal ')
    output_file.write('sample[] = {')
    for i in fsum:
        output_file.write(str(i))
        if (i != fsum[len(fsum) - 1]):
            output_file.write(', ')
    output_file.write('};')

# def as_python_var(fsum):


srate = 100
stime = 10
t = np.linspace(0, stime, stime*srate)

f1 = np.sin(2*np.pi*5*t)
f2 = np.sin(2*np.pi*3*t)
fsum = f1 + f2
as_cpp_var(fsum, 'cufft')

Y = np.fft.fft(fsum)
amplitude = 2*np.abs(Y)/len(t)

# hz between 0 and Nyquist
hz = np.linspace(0, np.floor(srate/2), np.floor(stime*srate/2) + 1)

plt.stem(hz, amplitude[0:len(hz)])
plt.show()
