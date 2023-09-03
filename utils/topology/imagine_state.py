from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def func_null(state):
    return 10

def jac_null(state):
    return np.zeros((128,))

def func(state):
    state = state.reshape((64,2))
    centers = (state[1:,:]+state[:-1,:])/2.0
    norms = np.sum(centers*centers, axis=1)
    dists = norms[:,np.newaxis] + norms[np.newaxis,:] - 2*np.dot(centers, centers.transpose())
    dists = np.sqrt(np.maximum(dists, 0.0))
    loss = np.sum(1/np.maximum(dists, 0.005))
    loss -= 200*dists.shape[0]
    return loss

def jac(state):
    state = state.reshape((64,2))
    centers = (state[1:,:]+state[:-1,:])/2.0
    norms = np.sum(centers*centers, axis=1)
    dists = norms[:,np.newaxis] + norms[np.newaxis,:] - 2*np.dot(centers, centers.transpose())
    dists = np.sqrt(np.maximum(dists, 0.0))
    g_dists = -1/(dists*dists)
    g_dists = np.where(dists > 0.005, g_dists, 0.0)
    diff = centers[:,np.newaxis,:]-centers[np.newaxis,:,:]
    p_dists_p_center = diff / (dists[:,:,np.newaxis]+1e-6)

    g_center = np.sum(p_dists_p_center*g_dists[:,:,np.newaxis], axis=1) - np.sum(p_dists_p_center*g_dists[:,:,np.newaxis], axis=0)
    g_state = np.zeros((64,2))
    g_state[1:,:] += g_center / 2.0
    g_state[:-1,:] += g_center/ 2.0
    return g_state.flatten()

def constr_func(state, i, j, l0):
    state = state.reshape((64,2))
    diff = state[i,:]-state[j,:]
    dist = np.sqrt(np.sum(diff*diff))
    return dist-l0

def constr_jac(state, i, j, l0):
    state = state.reshape((64,2))
    diff = state[i,:]-state[j,:]
    dist = np.sqrt(np.sum(diff*diff))
    grad = diff / dist
    g_state = np.zeros((64,2))
    g_state[i,:] += grad
    g_state[j,:] -= grad
    return g_state.flatten()

# test func and jac consistancy
#x0 = np.zeros((128,))
#num_g = np.zeros((128,))
#x0[0::2] = np.linspace(0,1,64)
#for i in range(127):
#    x0[i]+=0.0001
#    l1=func(x0)
#    x0[i]-=0.0002
#    l2=func(x0)
#    x0[i]+=0.0001
#    num_g[i]=(l1-l2)/0.0002
#jac_g = jac(x0)
#print(num_g)
#print(jac_g)
#pdb.set_trace()


NUM_POINTS = 64
L0 = 0.016
def setup_constraints(intersections):
    # return a list of dictionary of constraints to be used by SLSQP
    # intersections is a list of tuples of node index.
    constraints = []
    for i in range(NUM_POINTS-1):
        constr = {'type':'eq',
                  'fun':constr_func,
                  'jac':constr_jac,
                  'args':[i,i+1,L0]}
        constraints.append(constr)
    for i,j in intersections:
        constr = {'type':'eq',
                  'fun':constr_func,
                  'jac':constr_jac,
                  'args':[i,i+1,0.0]}
        constraints.append(constr)
    return constraints

# does not satisfy constraints. does it work?
x0 = np.zeros((128,))
x0[0::2] = np.linspace(0,1,64)
x0[1::2] = np.linspace(0,3,64)
x0[1::2] = np.sin(x0[1::2])*0.01
#x0 = np.random.rand(128)
constraints = setup_constraints([(10,50)])
res = minimize(func, x0, method='COBYLA', jac=jac, constraints = constraints, options={'maxiter':5000})
x1 = res.x
if not res.success:
    print(res)
    raise RuntimeError("constraint cannot be satisfied")
res = minimize(func, x1, method='SLSQP', jac=jac, constraints = constraints, options={'maxiter':5000})
if res.success:
    x2 = res.x.reshape((64,2))
    plt.plot(x2[:,0],x2[:,1])
    plt.show()
else:
    print("optimization failed")

