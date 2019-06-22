import math
import numpy as np

t = np.linspace(0, 10, 100)

f1 = [5*math.sin(i) for i in t]

f2 = [3*math.sin(i+math.pi/3) for i in t]

fsum = [i + j for i, j in zip(f1, f2)]

output_file = open('time_data.txt', 'a')
output_file.write(', '.join(str(i) for i in fsum))
output_file.close()
