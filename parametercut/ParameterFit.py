import numpy as np
import csv
import matplotlib.pyplot as plt

with open('parameterfit_2l.txt','r') as f:
    f_iter = csv.reader(f)
    raw = [i for i in f_iter]
raw = np.array(raw[1:],dtype=float)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(raw[:,0], raw[:,1], raw[:,2])
ax.set_xlabel('upsmcmass')
ax.set_ylabel('v0mcmasssum')
ax.set_zlabel('massdiff')
ax.view_init(azim = 20,elev = 20)
