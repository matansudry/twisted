import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

root = sys.argv[1]
for i in range(1,77):
    state=np.loadtxt(root+'/node_position_%d.txt'%(i))
    x=state[:,0].reshape((2,64))
    y=state[:,1].reshape((2,64))
    z=state[:,2].reshape((2,64))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,z,ccount=70)
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.1,0.2)
    ax.view_init(elev=70, azim=-90)
    plt.savefig('%03d.png'%(i))
    plt.close()
