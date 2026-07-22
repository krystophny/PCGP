import os
from jinja2 import Template
from sympy.printing.pycode import pycode
import sympy
from ..symbolic_kernels import symbolic_mercer_kernel, symbolic_parametrization_kernel
import inspect
from dataclasses import dataclass
import numpy as np


KERNEL_TEMPLATE = Template("""
import jax.numpy as jnp

                      
{% for body, parameters_of_kernel in kernel_inputs %}    
def kernel_{{ loop.index0 }}(x1, x2, parameters, structure): 
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = jnp.zeros((x1.shape[0], x2.shape[0]))
{{ body}}
    for i, idxs1 in enumerate(structure.x1):
        for j, idxs2 in enumerate(structure.x2):
            x_block = xx[idxs1]
            y_block = yy[idxs2]
            block = function_grid[i][j](x_block[:, None, :], y_block[None, :, :])
            K = K.at[jnp.ix_(idxs1, idxs2)].set(block)

    return K
{% endfor %}

def kernel(x1, x2, parameters, structure):
    k = ({% for i in range(number_of_kernels) %}
        kernel_{{ loop.index0 }}(x1, x2, parameters, structure)
        {% if not loop.last %} + {% endif %}{% endfor %}) 
    return k
                
""")

@dataclass
class KernelSpecifics:
    """
    Dataclass containing necessary kernel-specific information used in the Template.

    ----------
    body : str
        The body of the kernel function, which includes the definitions of the kernel elements and the construction of the function grid.
    parameters : list[str]
    The parameters used in this kernel
    input_dims : int
        The number of input dimensions (excluding the task label) used in this kernel.
    num_tasks : int
        The number of tasks in the multi-task setting, which determines the size of the function grid. 
    """
    body: str
    parameters: list[str]
    input_dims: int
    num_tasks: int
    

class PCGP_Builder_jax:
    """
    Writes a file containing all kernel functions and the main kernel function, which is a sum of the individual added kernels. 
    The kernel functions are generated based on the user provided input to the method add_kernel. 
    The generated file can be imported and used in a JAX-based Gaussian process model.
    """
    def __init__(self):
        self.kernels = []
    
    def add_kernel(
        self,
        input_matrix,
        number_of_input_dimensions = 1,
        mercer = False,
        Sigma = None,
        base_kernel = None,
        base_kernel_arguments = None,
        shared_base_kernel = False
        ):
        """
        Adds new kernel to total additive kernel according to user-provided specifications.
        
        :param input_matrix: function returning a sympy.matrices.Matrix.
            If mercer is False, the matrix is interpreted as the operator matrix applied to the base kernel. The function takes two arguments, the first is a list of (symbolic) differential operators and the second is a list of (symbolic) input variables. Physical parameters must be sympy.symbols.
            Example: 
                def B(D, x):
                    a = sp.Symbol("a")
                    return sp.matrices.Matrix([[1],
                                                [a*D[0]]])
            If mercer is True, the columns of the matrix are interpreted as the (symbolic) base functions out of which the kernel is 
            constructed using Mercer's theorem. The function takes one argument, a list of (symbolic) input variables. Physical parameters must be sympy.symbols.
            Example:
                def B(x):
                    a = sp.Symbol("a")
                    return sp.matrices.Matrix([[sp.cos(x[0])],
                                                [sp.sin(a*x[0])]])
        :param number_of_input_dimensions: int
            Number of input dimensions (excluding the task label) used in this kernel. Either this or base_kernel_arguments needs to be specified. 
        :param mercer: Bool
            Flag. If False, the kernel is generated based on the operator matrix interpretation of input_matrix. If True, the kernel is generated based on the Mercer's theorem interpretation of input_matrix.
        :param Sigma: sp.matrices.Matrix, optional
            Only used if mercer is True. The postive semi-definite matrix Sigma in Mercer's theorem, which specifies correlation between base functions
        :param base_kernel: function, optional
            Only used if mercer is False. A positive semi-definite and symmetric function of x and y, returning a sympy object for symbolic input. If not specified, RBF kernel is used.
        :param base_kernel_arguments: list of strings, optional
            Only used if mercer is False. A list of strings containing expressions with indexed inputs of the form "x[0]", which specify the arguments of the base kernel. Either this or number_of_input_dimensions needs to be specified. If not specified, it is assumed that the base kernel takes all input dimensions as arguments, in the form "x[0]", "x[1]", etc.
        :param shared_base_kernel: Bool
            Flag, only used if mercer is False. If True, the same base kernel parameters are shared across all columns of input_matrix. If False, each parametrization vector gets its own set of base kernel parameters.
        """
        spec = self._generate_kernel_specifics(
            input_matrix,
            number_of_input_dimensions = number_of_input_dimensions,
            mercer = mercer,
            Sigma=Sigma,
            base_kernel = base_kernel,
            base_kernel_arguments = base_kernel_arguments,
            shared_base_kernel=shared_base_kernel
        )
        self.kernels.append(spec)

    @property
    def input_dims(self):
        return max(k.input_dims for k in self.kernels)

    @property
    def num_tasks(self):
        values = {k.num_tasks for k in self.kernels}
        if len(values) != 1:
            raise ValueError("Inconsistent num_tasks across kernels")
        return values.pop()
    
    @property 
    def parameters(self):
        params = set()
        for k in self.kernels:
            params.update(k.parameters)
        return sorted(params)
    
    def _generate_kernel_specifics(
        self,
        input_matrix,
        number_of_input_dimensions=None,
        mercer=False,
        Sigma = None,
        base_kernel=None,
        base_kernel_arguments=None,
        shared_base_kernel = False) -> KernelSpecifics:
        
        if base_kernel_arguments:
            if number_of_input_dimensions and number_of_input_dimensions != len(base_kernel_arguments):
                raise ValueError("number_of_input_dimensions does not match length of base_kernel_arguments")
            eff_input_dims = len(base_kernel_arguments)
        else:
            if not number_of_input_dimensions:
                raise ValueError("Either number_of_input_dimensions or base_kernel_arguments must be provided")
            eff_input_dims = number_of_input_dimensions

        if mercer:
            kernel_object = symbolic_mercer_kernel(input_matrix, Sigma = Sigma, number_of_input_dimensions=number_of_input_dimensions) 
        else: 
            kernel_object = symbolic_parametrization_kernel(input_matrix, base_kernel = base_kernel, eff_input_dims = eff_input_dims, shared_base_kernel=shared_base_kernel) 
        symbolic_kernel = kernel_object.get_symbolic_kernel()
        num_tasks = symbolic_kernel.shape[0]

        return KernelSpecifics(
            body=self._get_body(kernel_object, base_kernel_arguments, symbolic_kernel, num_tasks, number_of_input_dimensions),
            parameters=kernel_object.parameters,
            input_dims=number_of_input_dimensions,
            num_tasks=num_tasks,
       )
    
    def _get_body(self, kernel_object, base_kernel_arguments, symbolic_kernel, num_tasks, number_of_input_dimensions):
        parameters = kernel_object.parameters
        lines = []
        parameter_lines = []
        for p in parameters:
            parameter_lines.append(f"{p} = parameters['{p}']")
        lines.extend(parameter_lines)        

        substitutions = {"x": sympy.IndexedBase("x"), "y": sympy.IndexedBase("y")}

        #get number of input dimensions 
        if base_kernel_arguments: 
            mapping = {}
            number_of_input_dimensions = 1
            for k, arg_str in enumerate(base_kernel_arguments):
                target_expr_x = sympy.parsing.sympy_parser.parse_expr(arg_str, local_dict=substitutions)
                target_expr_y = target_expr_x.xreplace({substitutions["x"]: substitutions["y"]})
                mapping[sympy.Symbol(f"x{k+1}")] = target_expr_x
                mapping[sympy.Symbol(f"y{k+1}")] = target_expr_y
                argument_indices = [idx.indices[0] for idx in target_expr_x.atoms(sympy.Indexed)]
                if argument_indices:
                    number_of_input_dimensions = max(number_of_input_dimensions, max(argument_indices)+1)
        else:
            mapping = {sympy.Symbol(f"x{k+1}"): substitutions["x"][k] for k in range(number_of_input_dimensions)}
            mapping.update({sympy.Symbol(f"y{k+1}"): substitutions["y"][k] for k in range(number_of_input_dimensions)})


        #make jnp expressions for each kernel element
        for (i, j), element in np.ndenumerate(symbolic_kernel):
            expr = element.xreplace(mapping).simplify()
            jnp_expr = pycode(expr).replace("math.", "jnp.")
            for k in range(number_of_input_dimensions):
                jnp_expr = jnp_expr.replace(f"x[{k}]", f"x[...,{k}]")
                jnp_expr = jnp_expr.replace(f"y[{k}]", f"y[...,{k}]")
            if expr == 0:
                    jnp_expr = "jnp.zeros_like(x[...,0]-y[...,0])"
            lines.append(f"def k{i}{j}(x, y):")
            lines.append(f"    return {jnp_expr}")
    
        k_fns = ["function_grid = ["]+[f"    [{', '.join(f'k{i}{j}' for j in range(num_tasks))}]," for i in range(num_tasks)]+["]"]
        lines.extend(k_fns) 
        return "\n".join([" " * 4 + l for l in lines]) 
    
    def _get_sampling(self):
        lines = ["sampled_parameters = {}",
                "if priors is None: ",
                "   print('No prior specified, using default Uniform(0.001, 1) prior for all parameters.')",
                "   priors = {}"]
        for key in self.parameters:    
            lines.append(f"   priors['{key}'] = dist.Uniform(0.001, 1)") #arbitrary, can be changed in main!!
        lines.append("for key in priors:")
        lines.append("   sampled_parameters[key] = numpyro.sample(key, priors[key])")
        return "\n".join([" " * 4 + l for l in lines]) 

    def write(self, file_name, output_dir=None):
        """
        Writes file containing kernel functions.
        
        :param file_name: str
            name of generated file
        :param output_dir: filepath
            Path where file should be generated (especially necessary if used with jupyter notebooks). If not specified, file is generated in current folder.
        """
        rendered = KERNEL_TEMPLATE.render(
                class_name=file_name,
                kernel_inputs=[
                    (k.body, k.parameters)
                    for k in self.kernels
                ],
                number_of_kernels=len(self.kernels),
                number_of_input_dimensions=self.input_dims,
                num_tasks=self.num_tasks,
                sampling = self._get_sampling(),
                
            )
        if output_dir is None:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            output_dir = os.path.dirname(os.path.abspath(caller_file))
        file_path = os.path.join(output_dir, f"{file_name}.py")
        with open(file_path, "w") as f:
            f.write(rendered)
        print(f"Kernel and model written to: {file_path}")


