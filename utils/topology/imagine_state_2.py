import scipy.optimize as optimize
import numpy as np
import functools
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

def constr_func(state, pairs):
    # pairs is a list of tuples of node indices
    dists = []
    state = state.reshape((64,2))
    for i,j in pairs:
        diff = state[i,:]-state[j,:]
        dist = np.sqrt(np.sum(diff*diff))
        dists.append(dist)
    return np.array(dists)

def constr_jac(state, pairs):
    grads = []
    state = state.reshape((64,2))
    for i,j in pairs:
        diff = state[i,:]-state[j,:]
        dist = np.sqrt(np.sum(diff*diff))
        grad = diff / dist
        g_state = np.zeros((64,2))
        g_state[i,:] += grad
        g_state[j,:] -= grad
        grads.append(g_state.flatten())
    return np.array(grads)


NUM_POINTS = 64
L0 = 0.016
def setup_constraints(intersections):
    # return a list of dictionary of constraints to be used by SLSQP
    # intersections is a list of tuples of node index.
    constraints = []
    pairs = []
    for i in range(NUM_POINTS-1):
        pairs.append((i,i+1))
    constr = optimize.NonlinearConstraint(functools.partial(constr_func, pairs=pairs),
                                 L0, L0,
                                 jac=functools.partial(constr_jac, pairs=pairs))
    constraints.append(constr)
    pairs = []
    for i,j in intersections:
        pairs.append((i,j))
    constr = optimize.NonlinearConstraint(functools.partial(constr_func, pairs=pairs),
                                 0.0, 0.0,
                                 jac=functools.partial(constr_jac, pairs=pairs))

    constraints.append(constr)
    return constraints

# does not satisfy constraints. does it work?
x0 = np.zeros((128,))
x=np.linspace(0,5,64)
x0[0::2] = np.cos(x)*0.2
x0[1::2] = np.sin(x)*0.2
plt.scatter(x0[0::2], x0[1::2], c=np.arange(64))
plt.axis('equal')
plt.show()
#x0 = np.random.rand(128)
constraints = setup_constraints([(10,50)])
res = optimize.minimize(func, x0, method='trust-constr', jac=jac, constraints = constraints, options={'maxiter':3000})
x1 = res.x
if not res.success:
    print(res)
    raise RuntimeError("constraint cannot be satisfied")
#res = optimize.minimize(func, x1, method='SLSQP', jac=jac, constraints = constraints, options={'maxiter':5000})
if res.success:
    x2 = res.x.reshape((64,2))
    plt.scatter(x2[:,0],x2[:,1],c=np.arange(64))
    plt.axis('equal')
    plt.show()
else:
    print("optimization failed")

