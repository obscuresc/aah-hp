import math
import numpy as np

t = np.linspace(0, 10, 100)

f1 = 5*sin(t)
f2 = 3*sin(t+pi/3)

fsum = f1 + f2

file = open('time_data.txt', 'a')
f.write(fsum)
f.close()
