
import torch
import gpytorch
from einops import rearrange
from PCGP import ConstraintsModifications


class PCGP_Kernel_0(gpytorch.kernels.Kernel):
    def __init__(self, parameter_modifications = {}, number_of_input_dimensions=1, num_tasks=3, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.parameters = {'amplitude': None, 'lengthscale': None}
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
        x1_data = x1[:,:-1] #the last column of the input tensors is expected to contain the task indices, the rest are data dimensions
        i1 = x1[:,-1] 
        x2_data = x2[:,:-1]
        i2 = x2[:,-1]
        # ensure correct shapes
        if x1_data.dim() == 1:
            x1_data = x1_data.unsqueeze(-1)
        if x2_data.dim() == 1:
            x2_data = x2_data.unsqueeze(-1)
        # for broadcasting 
        x1_ = x1_data.unsqueeze(-2)   # (N, 1, D) -> broadcasts to (N, M, D)
        x2_ = x2_data.unsqueeze(-3)   # (1, M, D)
        i1_ = i1.unsqueeze(-1)        # (N, 1)
        i2_ = i2.unsqueeze(-2)        # (1, M)
        amplitude = self.get_param('amplitude')
        lengthscale = self.get_param('lengthscale')
        def k00_fn(x, y):
            return amplitude*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
        def k01_fn(x, y):
            return amplitude*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
        def k02_fn(x, y):
            return amplitude*(lengthscale**2 - lengthscale + (x[...,0] - y[...,0])**2)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
        def k10_fn(x, y):
            return amplitude*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
        def k11_fn(x, y):
            return amplitude*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
        def k12_fn(x, y):
            return amplitude*(lengthscale**2 - lengthscale + (x[...,0] - y[...,0])**2)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
        def k20_fn(x, y):
            return amplitude*(lengthscale**2 - lengthscale + (x[...,0] - y[...,0])**2)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
        def k21_fn(x, y):
            return amplitude*(lengthscale**2 - lengthscale + (x[...,0] - y[...,0])**2)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
        def k22_fn(x, y):
            return amplitude*(lengthscale**4 + 2*lengthscale**2*(-lengthscale + (x[...,0] - y[...,0])**2) + 2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**4
        k_fns = {
            (0, 0): k00_fn,
            (0, 1): k01_fn,
            (0, 2): k02_fn,
            (1, 0): k10_fn,
            (1, 1): k11_fn,
            (1, 2): k12_fn,
            (2, 0): k20_fn,
            (2, 1): k21_fn,
            (2, 2): k22_fn,
        }
        # assemble kernel
        K = 0
        for (t1, t2), fn in k_fns.items():
            mask = (i1_ == t1) & (i2_ == t2)   # call the right kernel function for each task pair
            K = K + fn(x1_, x2_) * mask
        if diag:
            return torch.diag(K)
        return K

class PCGP_Kernel_1(gpytorch.kernels.Kernel):
    def __init__(self, parameter_modifications = {}, number_of_input_dimensions=1, num_tasks=3, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.parameters = {}
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
        x1_data = x1[:,:-1] #the last column of the input tensors is expected to contain the task indices, the rest are data dimensions
        i1 = x1[:,-1] 
        x2_data = x2[:,:-1]
        i2 = x2[:,-1]
        # ensure correct shapes
        if x1_data.dim() == 1:
            x1_data = x1_data.unsqueeze(-1)
        if x2_data.dim() == 1:
            x2_data = x2_data.unsqueeze(-1)
        # for broadcasting 
        x1_ = x1_data.unsqueeze(-2)   # (N, 1, D) -> broadcasts to (N, M, D)
        x2_ = x2_data.unsqueeze(-3)   # (1, M, D)
        i1_ = i1.unsqueeze(-1)        # (N, 1)
        i2_ = i2.unsqueeze(-2)        # (1, M)
        def k00_fn(x, y):
            return torch.cos(x[...,0] - y[...,0])
        def k01_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        def k02_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        def k10_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        def k11_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        def k12_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        def k20_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        def k21_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        def k22_fn(x, y):
            return torch.zeros_like(x[...,0], device=x.device)
        k_fns = {
            (0, 0): k00_fn,
            (0, 1): k01_fn,
            (0, 2): k02_fn,
            (1, 0): k10_fn,
            (1, 1): k11_fn,
            (1, 2): k12_fn,
            (2, 0): k20_fn,
            (2, 1): k21_fn,
            (2, 2): k22_fn,
        }
        # assemble kernel
        K = 0
        for (t1, t2), fn in k_fns.items():
            mask = (i1_ == t1) & (i2_ == t2)   # call the right kernel function for each task pair
            K = K + fn(x1_, x2_) * mask
        if diag:
            return torch.diag(K)
        return K


class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  parameter_modifications = {}, number_of_input_dimensions = 1, num_tasks = 3, priors = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.num_tasks = num_tasks
        self.number_of_input_dimensions = number_of_input_dimensions
        self.covar_module = (
                            PCGP_Kernel_0(parameter_modifications, number_of_input_dimensions=self.number_of_input_dimensions, num_tasks=self.num_tasks)
                             + 
                            PCGP_Kernel_1(parameter_modifications, number_of_input_dimensions=self.number_of_input_dimensions, num_tasks=self.num_tasks)
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
        mean_x = self.mean_module(x[:,0]) ##no need for task specific mean since it's zero
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 