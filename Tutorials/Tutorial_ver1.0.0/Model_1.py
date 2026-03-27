
import torch
import gpytorch
from einops import rearrange
from PCGP import ConstraintsModifications

class PCGP_Kernel(gpytorch.kernels.Kernel):
    def __init__(self, input_parameters, number_of_input_dimensions=1, num_tasks=2, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.input_parameters = input_parameters
        self.param_constraints = {}
        self.number_of_input_dimensions = number_of_input_dimensions

        for param_name in input_parameters:
            raw_name = f"raw_{param_name}"
            value, requires_grad, constraint = input_parameters[param_name]
            param = torch.nn.Parameter(torch.tensor([value]), requires_grad=requires_grad)
            self.register_parameter(raw_name, param)
            if constraint and requires_grad:
                CM = ConstraintsModifications(constraint)
                init_val = value if CM.is_fulfilled(value) else CM.init_val_from_constraint()
                self.register_constraint(raw_name, constraint)
                self.param_constraints[param_name] = constraint
                self.set_param(param_name, init_val)

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
        if param_name in self.param_constraints:
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
            xx, yy = mesh[0].T.unsqueeze(0), mesh[1].T.unsqueeze(0)
        elif x1.dim() == 2 and x1.shape[1] == self.number_of_input_dimensions:
            xx = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)
            yy = torch.zeros((self.number_of_input_dimensions, x1.shape[0], x2.shape[0]), device=x1.device)
            for i in range(self.number_of_input_dimensions):
                mesh = torch.meshgrid(torch.squeeze(x1[:,i]), torch.squeeze(x2[:,i]), indexing='xy')
                xx[i] = mesh[0].T
                yy[i] = mesh[1].T
        R = self.get_param('R')
        A = self.get_param('A')
        l = self.get_param('l')
        k00 = A*torch.exp(-1/2*(xx[0] - yy[0])**2/l)
        k01 = A*(R*(xx[0] - yy[0]) + l)*torch.exp(-1/2*(xx[0] - yy[0])**2/l)/l
        k10 = A*(-R*(xx[0] - yy[0]) + l)*torch.exp(-1/2*(xx[0] - yy[0])**2/l)/l
        k11 = A*(R**2*(l - (xx[0] - yy[0])**2) + l**2)*torch.exp(-1/2*(xx[0] - yy[0])**2/l)/l**2
        cov_m = torch.squeeze(torch.cat([
    torch.cat([k00, k01], dim=-1),
    torch.cat([k10, k11], dim=-1)], dim=-2))
        cov_f = rearrange(cov_m, "(t1 w1) (t2 w2)-> (w1 t1) (w2 t2)", t1=2, t2=2)
        return torch.diag(cov_f) if diag else cov_f


class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  model_parameters, number_of_input_dimensions = 1, num_tasks = 2, priors = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks 
        )
        self.num_tasks = num_tasks
        self.number_of_input_dimensions = number_of_input_dimensions
        self.covar_module = PCGP_Kernel(model_parameters, number_of_input_dimensions=self.number_of_input_dimensions, num_tasks = self.num_tasks)
        
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