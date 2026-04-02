from jinja2 import Template
import os
import inspect
import sympy as sp
import numpy as np
from sympy.printing.pycode import pycode
from dataclasses import dataclass
from .symbolic_kernels import symbolic_mercer_kernel, symbolic_parametrization_kernel




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
        self.parameters = {{parameters_of_kernel}}
        self.param_constraints = {}
        self.number_of_input_dimensions = number_of_input_dimensions
                           
        for param_name in self.parameters:
            raw_name = f"raw_{param_name}"
            param = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.register_parameter(raw_name, param)   
            if param_name[:-2] == "amplitude" or param_name[:-2] == "lengthscale":
                self.register_constraint(raw_name, gpytorch.constraints.Positive())
                self.param_constraints[param_name] = gpytorch.constraints.Positive() 
                           
        for param_name, (value, requires_grad, constraint) in parameter_modifications.items():
            if param_name in self.parameters:
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
        if param_name in self.parameters:
            self._set_param(param_name, value)
        else:
            raw_name = f"raw_{param_name}"
            value_tensor = torch.nn.Parameter(torch.tensor([value]))
            self.register_parameter(raw_name, value_tensor)

    #def num_outputs_per_input(self, x1, x2):
    #    return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
{{ body }}
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
            for name, prior in priors.items():
                self.register_prior(
                    name+"_prior",
                    prior,
                    lambda m: m.covar_module.get_param(name),
                    lambda m, val: m.covar_module._set_param(name, val),
                )

    def forward(self, x):
        mean_x = self.mean_module(x[:,0]) ##no need for task specific mean since it's zero
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
    def __init__(self):
        print("You are using the new version of the generator, which is still experimental. If you encounter any issues, please report them to the developer.")
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
        lines = [
            "x1_data = x1[:,:-1] #the last column of the input tensors is expected to contain the task indices, the rest are data dimensions",
            "i1 = x1[:,-1] ",
            "x2_data = x2[:,:-1]",
            "i2 = x2[:,-1]",

            "# ensure correct shapes",
            "if x1_data.dim() == 1:",
            "    x1_data = x1_data.unsqueeze(-1)",
            "if x2_data.dim() == 1:",
            "    x2_data = x2_data.unsqueeze(-1)",

            "# for broadcasting ",
            "x1_ = x1_data.unsqueeze(-2)   # (N, 1, D) -> broadcasts to (N, M, D)",
            "x2_ = x2_data.unsqueeze(-3)   # (1, M, D)",
            "i1_ = i1.unsqueeze(-1)        # (N, 1)",
            "i2_ = i2.unsqueeze(-2)        # (1, M)",
        ]    
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
                torch_expr = "torch.zeros_like(x[...,0], device=x.device)"
            lines.append(f"def k{i}{j}_fn(x, y):")
            lines.append(f"    return {torch_expr}")

        
        dict_rows = ["k_fns = {"]
        for i in range(num_tasks):
            for j in range(num_tasks):
                dict_rows.append(f"    ({i}, {j}): k{i}{j}_fn,")
        dict_rows.append("}")
        lines.extend(dict_rows)


        assembling_kernel = [
        "# assemble kernel",
        "K = 0",
        "for (t1, t2), fn in k_fns.items():",
        "    mask = (i1_ == t1) & (i2_ == t2)   # call the right kernel function for each task pair",
        "    K = K + fn(x1_, x2_) * mask",
        "if diag:",
        "    return torch.diag(K)",
        "return K"]
        lines.extend(assembling_kernel)
        return "\n".join([" " * 8 + l for l in lines])
    
    
    def write(self, class_name, output_dir=None):
        rendered = KERNEL_TEMPLATE.render(
            class_name=class_name,
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
        file_path = os.path.join(output_dir, f"{class_name}.py")
        with open(file_path, "w") as f:
            f.write(rendered)
        print(f"Kernel and model written to: {file_path}")


