#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
#%%
import numpy as np
import matplotlib.pyplot as plt
a_0 = -3.0876
a_1 = -3.557528
a_2 = 0.781027
a_3 = 1.021521
a_4 = -0.156012
a_5 = -0.444058
a_6 = 0.019977
a_7 = 0.086850
a_8 = -0.005874
a_9 = -0.006809
a_10 = 8.25*10**(-4)
a_11 = 5.54*10**(-5)

N_POINTS = 100
a_coeff = np.array([a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11])
distance = np.logspace(np.log10(0.01), np.log10(200), N_POINTS)

exponents = np.arange(12)
distance_matrix = distance[:, np.newaxis]*np.ones([N_POINTS, 12])
N = 10**np.sum(a_coeff * (np.log10(distance_matrix)**exponents), axis=1)

fig, ax = plt.subplots(1, figsize=(6, 14))
ax.scatter(distance, N)
ax.set(xscale='log', yscale='log', ylim=(1e-7, 1e4), xlim=(0.001, 300))
plt.xlabel('Diameter of craters [km]')
plt.ylabel('Cumulative Crater Frequency [per sq.Km]')

