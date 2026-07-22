from jinja2 import Template
import os
import inspect
import sympy as sp
import numpy as np
from sympy.printing.pycode import pycode
from dataclasses import dataclass
from ..symbolic_kernels import symbolic_mercer_kernel, symbolic_parametrization_kernel




KERNEL_TEMPLATE = Template("""
import torch
import gpytorch
from einops import rearrange
from PCGP import ConstraintsModifications

{% for body, parameters_of_kernel in kernel_inputs %}
class PCGP_Kernel_{{ loop.index0 }}(gpytorch.kernels.Kernel):
    def __init__(self, parameter_modifications = {}, number_of_input_dimensions={{ input_dims }}, num_tasks={{ num_tasks }}, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.parameter_dict = {{parameters_of_kernel}}
        self.param_constraints = {}
        self.number_of_input_dimensions = number_of_input_dimensions
        unknown_parameters = sorted(set(parameter_modifications) - set(self.parameter_dict))
        if unknown_parameters:
            raise ValueError(
                f"Unknown parameter names: {unknown_parameters}. "
                f"Valid names: {sorted(self.parameter_dict)}"
            )
                           
        for param_name in self.parameter_dict:
            raw_name = f"raw_{param_name}"
            param = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.register_parameter(raw_name, param)   
            if param_name[:-2] == "amplitude" or param_name[:-2] == "lengthscale":
                self.register_constraint(raw_name, gpytorch.constraints.Positive())
                self.param_constraints[param_name] = gpytorch.constraints.Positive() 
                           
        for param_name, (value, requires_grad, constraint) in parameter_modifications.items():
            if param_name in self.parameter_dict:
                raw_name = f"raw_{param_name}"
                raw_param = getattr(self, raw_name)
                # handle constraints
                if constraint and requires_grad:
                    CM = ConstraintsModifications(constraint)
                    value = value if CM.is_fulfilled(value) else CM.init_val_from_constraint()
                    self.register_constraint(raw_name, constraint)
                    self.param_constraints[param_name] = constraint
                self.set_param(param_name, value)   
                raw_param.requires_grad_(requires_grad)

    def _set_param(self, param_name, value):
        raw_name = f"raw_{param_name}"
        param = getattr(self, raw_name)
        value = torch.as_tensor(value).to(param)
        if param_name in self.param_constraints:
            value = self.param_constraints[param_name].inverse_transform(value)
        self.initialize(**{raw_name: value})

    def get_param(self, param_name):
        raw_name = f"raw_{param_name}"
        raw_param = getattr(self, raw_name)
        if param_name in self.param_constraints:
            return self.param_constraints[param_name].transform(raw_param)
        return raw_param

    def get_raw_param(self, param_name):
        return getattr(self, f"raw_{param_name}")

    def set_param(self, param_name, value):
        if param_name in self.parameter_dict:
            self._set_param(param_name, value)
        else:
            raise ValueError(
                f"Unknown parameter name: {param_name}. "
                f"Valid names: {sorted(self.parameter_dict)}"
            )


    def forward(self, x1, x2, diag=False, **params):
        N, M = x1.size(0), x2.size(0)
            # 1. Separate features from indices
            # We use .long() because indices must be integers for argsort/bincount
        idx1 = x1[:, -1].long() #indices must be integers for argsort/bincount"
        idx2 = x2[:, -1].long()
        data1 = x1[:, :-1]
        data2 = x2[:, :-1]
        if data1.dim() == 1: #make sure data is 2D to avoid dimension issues
            data1 = data1.unsqueeze(-1)
        if data2.dim() == 1:
            data2 = data2.unsqueeze(-1)
                    # 2. Sort indices and reorder data rows
        sort_idx1 = torch.argsort(idx1)
        sort_idx2 = torch.argsort(idx2)
            
        sorted_data1 = data1[sort_idx1]
        sorted_data2 = data2[sort_idx2]
            
            # 3. Get split sizes
        counts1 = torch.bincount(idx1).tolist()
        counts2 = torch.bincount(idx2).tolist()
        splits1 = torch.split(sorted_data1, counts1)
        splits2 = torch.split(sorted_data2, counts2)
{{ body }}
                           
        rows = []
        for i, s1 in enumerate(splits1):
            row_blocks = []
            s1_expanded = s1.unsqueeze(1)
            for j, s2 in enumerate(splits2):
                s2_expanded = s2.unsqueeze(0)
                block = function_grid[i][j](s1_expanded, s2_expanded)
                row_blocks.append(block)
            rows.append(torch.cat(row_blocks, dim=1))
        sorted_matrix = torch.cat(rows, dim=0)
        # assemble kernel
        K = torch.empty((N, M), device=x1.device, dtype=sorted_matrix.dtype)
        K[sort_idx1[:, None], sort_idx2] = sorted_matrix
        if diag:
            return torch.diag(K)
        return K
{% endfor %}

class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  parameter_modifications = {}, number_of_input_dimensions = {{ input_dims }}, num_tasks = {{ num_tasks }}, priors = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.num_tasks = num_tasks
        self.number_of_input_dimensions = number_of_input_dimensions
        self.covar_module = ({% for i in range(number_of_kernels) %}
                            PCGP_Kernel_{{i}}(parameter_modifications, number_of_input_dimensions=self.number_of_input_dimensions, num_tasks=self.num_tasks)
                            {% if not loop.last %} + {% endif %}{% endfor %})
                        
        if priors:
            if hasattr(self.covar_module, "parameter_dict"):
                valid_parameter_names = set(self.covar_module.parameter_dict)
            else:
                valid_parameter_names = {
                    name
                    for kernel in self.covar_module.kernels
                    for name in kernel.parameter_dict
                }
            unknown_priors = sorted(set(priors) - valid_parameter_names)
            if unknown_priors:
                raise ValueError(
                    f"Unknown prior parameter names: {unknown_priors}. "
                    f"Valid names: {sorted(valid_parameter_names)}"
                )
            for name, prior in priors.items():
                self.register_prior(
                    name+"_prior",
                    prior,
                    lambda m: m.covar_module.get_param(name),
                    lambda m, val: m.covar_module._set_param(name, val),
                )

    def forward(self, x):
        mean_x = self.mean_module(x[...,0]) ##no need for task specific mean since it's zero
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 
""")

    

@dataclass
class KernelSpecifics:
    body: str
    parameters: list[str]
    input_dims: int
    num_tasks: int

class PCGP_Builder:
    """
    Writes a file containing all kernel functions and the main kernel function, which is a sum of the individual added kernels. 
    The kernel functions are generated based on the user provided input to the method add_kernel. 
    The generated file can be imported and used in a GPyTorch-based Gaussian process model.
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
        lines = []    
        for p in kernel_object.parameters:
            lines.append(f"{p} = self.get_param('{p}')")

        substitutions = {"x": sp.IndexedBase("x"), "y": sp.IndexedBase("y")}

        #get number of input dimensions 
        if base_kernel_arguments: 
            mapping = {}
            number_of_input_dimensions = 1
            for k, arg_str in enumerate(base_kernel_arguments):
                target_expr_x = sp.parsing.sympy_parser.parse_expr(arg_str, local_dict=substitutions)
                target_expr_y = target_expr_x.xreplace({substitutions["x"]: substitutions["y"]})
                mapping[sp.Symbol(f"x{k+1}")] = target_expr_x
                mapping[sp.Symbol(f"y{k+1}")] = target_expr_y
                argument_indices = [idx.indices[0] for idx in target_expr_x.atoms(sp.Indexed)]
                if argument_indices:
                    number_of_input_dimensions = max(number_of_input_dimensions, max(argument_indices)+1)
        else:
            mapping = {sp.Symbol(f"x{k+1}"): substitutions["x"][k] for k in range(number_of_input_dimensions)}
            mapping.update({sp.Symbol(f"y{k+1}"): substitutions["y"][k] for k in range(number_of_input_dimensions)})

        #make torch expressions for each kernel element
        for (i, j), element in np.ndenumerate(symbolic_kernel):
            expr = element.xreplace(mapping).simplify()
            torch_expr = pycode(expr).replace("math.", "torch.")
            for k in range(number_of_input_dimensions):
                torch_expr = torch_expr.replace(f"x[{k}]", f"x[...,{k}]")
                torch_expr = torch_expr.replace(f"y[{k}]", f"y[...,{k}]")
            if expr == 0:
                torch_expr = "torch.zeros_like(x[...,0]-y[...,0], device=x.device)"
            lines.append(f"def k{i}{j}_fn(x, y):")
            lines.append(f"    return {torch_expr}")

        k_fns = ["function_grid = ["]+[f"    [{', '.join(f'k{i}{j}_fn' for j in range(num_tasks))}]," for i in range(num_tasks)]+["]"]
        lines.extend(k_fns) 
        return "\n".join([" " * 8 + l for l in lines])
    
    def write(self, file_name, output_dir=None):
        """
        Writes file containing GPyTorch-style kernel and model classes.
        
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
            input_dims=self.input_dims,
            num_tasks=self.num_tasks,
        )
        if output_dir is None:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            output_dir = os.path.dirname(os.path.abspath(caller_file))
        file_path = os.path.join(output_dir, f"{file_name}.py")
        with open(file_path, "w") as f:
            f.write(rendered)
        print(f"Kernel and model written to: {file_path}")

