
import torch
import gpytorch
from einops import rearrange
from PCGP import ConstraintsModifications


class PCGP_Kernel_0(gpytorch.kernels.Kernel):
    def __init__(self, parameter_modifications = {}, number_of_input_dimensions=1, num_tasks=2, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.parameters = {'lengthscale': None, 'R': None, 'amplitude': None}
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
        if x1.dim() == 1:
            mesh = torch.meshgrid(x1.flatten(), x2.flatten(), indexing='xy')
            x, y = mesh[0].T.unsqueeze(0), mesh[1].T.unsqueeze(0)
        elif x1.dim() == 2 and x1.shape[1] == self.number_of_input_dimensions:
            x = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)
            y = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)
            for i in range(self.number_of_input_dimensions):
                mesh = torch.meshgrid(torch.squeeze(x1[:,i]), torch.squeeze(x2[:,i]), indexing='xy')
                x[i] = mesh[0].T
                y[i] = mesh[1].T
        lengthscale = self.get_param('lengthscale')
        R = self.get_param('R')
        amplitude = self.get_param('amplitude')
        k00 = amplitude*torch.exp(-1/2*(x[0] - y[0])**2/lengthscale)
        k01 = amplitude*(R*(x[0] - y[0]) + lengthscale)*torch.exp(-1/2*(x[0] - y[0])**2/lengthscale)/lengthscale
        k10 = amplitude*(-R*(x[0] - y[0]) + lengthscale)*torch.exp(-1/2*(x[0] - y[0])**2/lengthscale)/lengthscale
        k11 = amplitude*(R**2*(lengthscale - (x[0] - y[0])**2) + lengthscale**2)*torch.exp(-1/2*(x[0] - y[0])**2/lengthscale)/lengthscale**2
        cov_m = torch.squeeze(torch.cat([
    torch.cat([k00, k01], dim=-1),
    torch.cat([k10, k11], dim=-1)], dim=-2))
        cov_f = rearrange(cov_m, "(t1 w1) (t2 w2)-> (w1 t1) (w2 t2)", t1=2, t2=2)
        return torch.diag(cov_f) if diag else cov_f


class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  parameter_modifications = {}, number_of_input_dimensions = 1, num_tasks = 2, priors = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks 
        )
        self.num_tasks = num_tasks
        self.number_of_input_dimensions = number_of_input_dimensions
        self.covar_module = (
                            PCGP_Kernel_0(parameter_modifications, number_of_input_dimensions=self.number_of_input_dimensions, num_tasks=self.num_tasks)
                            )
                        
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