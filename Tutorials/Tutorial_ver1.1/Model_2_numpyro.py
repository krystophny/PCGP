
import jax.numpy as jnp

                      
    
def kernel_0(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
    lengthscale = parameters['lengthscale']
    amplitude = parameters['amplitude']
    R = parameters['R']
    def k00(x, y):
        return amplitude*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
    def k01(x, y):
        return amplitude*(R*(x[...,0] - y[...,0]) + lengthscale)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale
    def k10(x, y):
        return -amplitude*(R*(x[...,0] - y[...,0]) - lengthscale)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale
    def k11(x, y):
        return amplitude*(R**2*(lengthscale - (x[...,0] - y[...,0])**2) + lengthscale**2)*jnp.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    function_grid = [
        [k00, k01],
        [k10, k11],
    ]
    for i, idxs1 in enumerate(structure.x1):
        for j, idxs2 in enumerate(structure.x2):
            x_block = xx[idxs1]
            y_block = yy[idxs2]
            block = function_grid[i][j](x_block[:, None, :], y_block[None, :, :])
            K = K.at[jnp.ix_(idxs1, idxs2)].set(block)

    return K


def kernel(x1, x2, parameters, structure):
    k = (
        kernel_0(x1, x2, parameters, structure)
        ) 
    return k
                