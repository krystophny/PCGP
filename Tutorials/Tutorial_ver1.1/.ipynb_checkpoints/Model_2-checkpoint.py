
import torch
import gpytorch
from einops import rearrange
from PCGP import ConstraintsModifications


class PCGP_Kernel_0(gpytorch.kernels.Kernel):
    def __init__(self, parameter_modifications = {}, number_of_input_dimensions=1, num_tasks=2, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.parameter_dict = {'lengthscale_0': None, 'amplitude_0': None, 'R': None}
        self.param_constraints = {}
        self.number_of_input_dimensions = number_of_input_dimensions
                           
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
            raw_name = f"raw_{param_name}"
            value_tensor = torch.nn.Parameter(torch.tensor([value]))
            self.register_parameter(raw_name, value_tensor)


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
        lengthscale_0 = self.get_param('lengthscale_0')
        amplitude_0 = self.get_param('amplitude_0')
        R = self.get_param('R')
        def k00_fn(x, y):
            return amplitude_0*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale_0)
        def k01_fn(x, y):
            return amplitude_0*(R*(x[...,0] - y[...,0]) + lengthscale_0)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale_0)/lengthscale_0
        def k10_fn(x, y):
            return -amplitude_0*(R*(x[...,0] - y[...,0]) - lengthscale_0)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale_0)/lengthscale_0
        def k11_fn(x, y):
            return amplitude_0*(R**2*(lengthscale_0 - (x[...,0] - y[...,0])**2) + lengthscale_0**2)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale_0)/lengthscale_0**2
        function_grid = [
            [k00_fn, k01_fn],
            [k10_fn, k11_fn],
        ]
                           
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


class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  parameter_modifications = {}, number_of_input_dimensions = 1, num_tasks = 2, priors = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
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
        mean_x = self.mean_module(x[...,0]) ##no need for task specific mean since it's zero
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 