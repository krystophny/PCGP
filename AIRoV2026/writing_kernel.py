from PCGP import PCGP_Builder
import sympy as sp


def B(X, t):
    return sp.matrices.Matrix([[1],
                                  [1],
                                  [X[0]**2+1],
                                  ])
def base_functions(x):
    return sp.matrices.Matrix([[sp.cos(x[0]), sp.sin(x[0])],
                               [0,0],
                               [0,0]
                               ])
builder = PCGP_Builder()
builder.add_kernel(B, number_of_input_dimensions = 1, shared_base_kernel = True)
builder.add_kernel(base_functions, number_of_input_dimensions=1, mercer = True)
builder.write("Bipendulum")