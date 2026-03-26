from jinja2 import Template
import os
import inspect
import sympy as sp
import numpy as np
from sympy.printing.pycode import pycode
from dataclasses import dataclass
from sympy.printing.pycode import pycode
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
            if param_name == "amplitude" or param_name == "lengthscale":
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

    def num_outputs_per_input(self, x1, x2):
        return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
{{ body }}
{% endfor %}

class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  parameter_modifications = {}, number_of_input_dimensions = {{ input_dims }}, num_tasks = {{ num_tasks }}, priors = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks 
        )
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
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved = True) 
""")

@dataclass
class KernelSpecifics:
    body: str
    parameters: list[str]
    input_dims: int
    num_tasks: int

class PCGP_Builder:
    def __init__(self):
        self.kernels = []
    
    def add_kernel(
        self,
        input_matrix,
        number_of_input_dimensions = 1,
        mercer = False,
        Sigma = None,
        base_kernel = None,
        base_kernel_arguments = None
        ):
        spec = self._generate_kernel_specifics(
            input_matrix,
            number_of_input_dimensions,
            mercer,
            Sigma,
            base_kernel,
            base_kernel_arguments,
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
        number_of_input_dimensions=1,
        mercer=False,
        Sigma = None,
        base_kernel=None,
        base_kernel_arguments=None,) -> KernelSpecifics:
        
        if mercer:
            kernel_object = symbolic_mercer_kernel(input_matrix, Sigma = Sigma, number_of_input_dimensions=number_of_input_dimensions) 
        else: 
            kernel_object = symbolic_parametrization_kernel(input_matrix, number_of_input_dimensions=number_of_input_dimensions, base_kernel = base_kernel, base_kernel_arguments = base_kernel_arguments) 
        symbolic_kernel = kernel_object.get_symbolic_kernel()
        num_tasks = symbolic_kernel.shape[0]
        lines = [
            "if x1.dim() == 1:",
            "    mesh = torch.meshgrid(x1.flatten(), x2.flatten(), indexing='xy')",
            "    x, y = mesh[0].T.unsqueeze(0), mesh[1].T.unsqueeze(0)",
            "elif x1.dim() == 2 and x1.shape[1] == self.number_of_input_dimensions:",
            "    x = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)",
            "    y = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)",
            "    for i in range(self.number_of_input_dimensions):",
            "        mesh = torch.meshgrid(torch.squeeze(x1[:,i]), torch.squeeze(x2[:,i]), indexing='xy')",
            "        x[i] = mesh[0].T",
            "        y[i] = mesh[1].T",
        ]    
        for p in kernel_object.parameters:
            lines.append(f"{p} = self.get_param('{p}')")

        substitutions = {"x": sp.IndexedBase("x"), "y": sp.IndexedBase("y")}
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

        for (i, j), element in np.ndenumerate(symbolic_kernel):
            expr = element.xreplace(mapping).simplify()
            torch_expr = pycode(expr).replace("math.", "torch.")
            if expr == 0:
                torch_expr = "torch.zeros_like(x[0], device=x.device)"
            lines.append(f"k{i}{j} = {torch_expr}")

        # Assemble the cov matrix
        cat_rows = []
        for i in range(num_tasks):
            row_expr = ", ".join([f"k{i}{j}" for j in range(num_tasks)])
            cat_rows.append(f"torch.cat([{row_expr}], dim=-1)")
        full_mat = "torch.cat([\n    " + ",\n    ".join(cat_rows) + "], dim=-2)" 

        lines.append("cov_m = torch.squeeze(" + full_mat + ")")
        lines.append(f"cov_f = rearrange(cov_m, \"(t1 w1) (t2 w2)-> (w1 t1) (w2 t2)\", t1={num_tasks}, t2={num_tasks})")
        lines.append("return torch.diag(cov_f) if diag else cov_f")        

        return KernelSpecifics(
            body="\n".join([" " * 8 + l for l in lines]),
            parameters=kernel_object.parameters,
            input_dims=number_of_input_dimensions,
            num_tasks=num_tasks,
        )
    
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

