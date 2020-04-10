#%%
# Draw histogram
from matplotlib import pyplot as plt
import pandas as pd
import statistics
import seaborn as sns
from scipy.stats import norm
import matplotlib.mlab as mlab
import numpy as np

#%%
# y_train = pd.read_csv('/home/waqas/projects/unet/crop_increased_otsu.txt',sep='\t')
# y_train['dia'] = y_train['dia']*50*0.00001
# # y_train = y_train[y_train['dia'] > 0.01]
# y_train = y_train.iloc[:,0].where(y_train.iloc[:,0] < 230)
# y_train.plot(kind='hist', bins=40, logy=True, logx=True, cumulative=True)
# sort = -np.sort(-y_train)
# distance = np.log(np.arange(0, len(y_train), 1))
# plt.scatter(np.log(sort), distance)
# plt.xlabel('Diameter of craters [km]')
# plt.ylabel('CSFD [km]')
# plt.title('Crater radius frequency for the training data')
# plt.ylim(-100, 1000)
# print(sort)
# # plt.savefig('hist.eps', format='eps')
# # plt.plot(pdf_x,pdf_y,'k--')
# plt.figure()
# %%
# Rough code
# import numpy as np
# import matplotlib.pyplot as plt
# a_0 = -3.0876
# a_1 = -3.557528
# a_2 = 0.781027
# a_3 = 1.021521
# a_4 = -0.156012
# a_5 = -0.444058
# a_6 = 0.019977
# a_7 = 0.086850
# a_8 = -0.005874
# a_9 = -0.006809
# a_10 = 8.25*10**(-4)
# a_11 = 5.54*10**(-5)



# y_train = pd.read_csv('/home/waqas/projects/unet/crop_increased_otsu.txt',sep='\t', index_col=None)

# # y_train = y_train[y_train['dia'] < 230]
# N_POINTS = len(y_train)

# # y_train = y_train.iloc[:,0].where(y_train.iloc[:,0] < 255)
# # # li_ = []
# # # for index, rows in y_train.iterrows():
# # #     li_.append(rows[0])
# y_train = y_train * 50/100000
# y_train = y_train.where(y_train > 0.01)
# y_train = y_train.dropna()

# y_train = y_train.iloc[:,0]
# y_train = np.asarray(y_train)

# for i in range(len(y_train)):
#     print(y_train[i])
# a_coeff = np.array([a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11])
# distance = np.logspace(np.log10(y_train.min()), np.log10(y_train.max()), N_POINTS)
# print(distance.shape, 'distance shape')

# d = np.array([np.log10(x) for x in y_train])
# distance = np.logspace(np.log10(min(y_train)), np.log10(max(y_train)), N_POINTS)

# exponents = np.arange(12)
# distance_matrix = distance[:, np.newaxis]*np.ones([N_POINTS, 12])
# N = 10**np.sum(a_coeff * (np.log10(distance_matrix)**exponents), axis=1)
# print(distance_matrix.shape)
# print(len(N))
# distance_n = np.logspace(np.log10(0.01), np.log10(200), N_POINTS)
# distance_matrix_n = distance_n[:, np.newaxis]*np.ones([N_POINTS, 12])
# N_n = 10**np.sum(a_coeff * (np.log10(distance_matrix_n)**exponents), axis=1)

# fig, ax = plt.subplots(1, figsize=(6, 14))
# ax.scatter(distance_n, N_n)
# ax.scatter(distance, N, c='#bcbd22')
# plt.loglog(d, d)
# ax.set(xscale='log', yscale='log', ylim=(1e-7, 1e4), xlim=(0.001, 100))
# plt.xlabel('Crater Diameter [km]')
# plt.ylabel('Cumulative Crater Frequency [per sq.Km]')
# plt.savefig('experiment.eps', format='eps')

#%%
# y_train = pd.read_csv('/home/waqas/projects/unet/crop_increased_otsu.txt',sep='\t', index_col=None)
# y_train = y_train[y_train['dia'] < 230]
# y_train['dia'] = y_train['dia']*50*0.00001
# y_train.to_csv('diameter.txt')
# df = y_train['dia'].value_counts()
# df.to_csv('delete.txt')
# x = [x for x in np.arange(1, 100, 1)]
# x = pd.Series(v for v in x)
# cs = np.log(y_train['dia'].value_counts())
# cs.shape


# df = pd.read_csv('/home/waqas/projects/unet/results/otsu/extracted_crater.txt',sep='\t', index_col=None)
# df2 = pd.read_csv('/home/waqas/projects/unet/results/crater_dino.txt',sep='\t', index_col=None)

# Age curves
df = pd.read_csv('/home/waqas/projects/unet/crater.txt',sep=',', index_col=None)
df2 = pd.read_csv('/home/waqas/projects/unet/crater_dino.txt',sep=',', index_col=None)

df.drop(['y', 'x'], axis=1, inplace=True)
df2.drop(['y', 'x'], axis=1, inplace=True)


df['dia'] = df.sort_values(by=['dia'], ascending=True)
plt.scatter(df['num'], df['dia'].cumsum())
df = df.sort_values(by=['dia'], ascending=True)

# df['dia'] = df['dia']-3
# df = df[df['dia'] > 1]
# df['dia'] = df['dia']*50*0.000001
# df = df['dia'].value_counts()

df2 = df2[df2['dia'] > 1]
df2['dia'] = df2['dia']*50*0.000001
df2 = df2['dia'].value_counts()

plt.loglog(df['dia'].cumsum(), df['num'], marker='.', linestyle='')
plt.loglog(df2['dia'].cumsum(), df2['num'], marker='.', linestyle='')

plt.xlabel('Crater Diameter [km]')
plt.ylabel('N[cum] per sq.km')
plt.savefig('csfd.eps', format='eps')
plt.show()

# %%
# df = pd.read_csv('results/otsu/extracted_crater.txt', sep='\t')
# df = df.sort_values(by=['dia'], ascending=True)
# df = df['dia'].value_counts()
# df.to_csv('yaba.txt')

df = pd.read_csv('test.txt', sep=',')
df.sort_values(by=['dia'], ascending=True)
