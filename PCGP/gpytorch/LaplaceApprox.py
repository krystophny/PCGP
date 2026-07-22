import torch
from .constraint_handling import ConstraintsModifications
from dataclasses import dataclass

@dataclass
class LaplaceResult:
    covariance: torch.Tensor
    hessian: torch.Tensor 
    MAP_estimates: torch.Tensor
    parameter_index: dict
    

def laplace_approx(parameters, model, loss):
    """
    Calculates the hessian of the -marginal log likelihood loss and its inverse.
    Returns a LaplaceResult containing the Hessian, its inverse covariance,
    the MAP estimates, and their parameter indices.
    
    :param parameters: dict
        dictionary of parameters with parameter name strings as keys and list containing [initial value(float), requires_grad(bool), constraints(False or gpytorch.constraints object)
    :param model: Gpytorch model in training mode
    :param loss: calculated loss (implicitly including graph used to calculate it to access gradients)
    """
    device = loss.device
    dtype = loss.dtype
    parameters_with_gradient, parameter_has_constraint = [], []
    for key in parameters:
        if model.covar_module.get_param(key).requires_grad:
            parameters_with_gradient.append(key)
            parameter_has_constraint.append(parameters[key][2])
       
    raw_hessian = torch.zeros((len(parameters_with_gradient), len(parameters_with_gradient)), device=device, dtype=dtype)
    raw_first_derivative = []
    inv_transformation_first_derivative = torch.ones(len(parameters_with_gradient), device=device, dtype=dtype)
    inv_transformation_second_derivative = torch.zeros(len(parameters_with_gradient), device=device, dtype=dtype)
    parameter_index = {}

    MAP_estimates = torch.stack([
        model.covar_module.get_param(k).detach().reshape(())
        for k in parameters_with_gradient
    ]) if parameters_with_gradient else torch.empty(0, device=device, dtype=dtype)

    if not parameters_with_gradient:
        empty_matrix = torch.empty((0, 0), device=device, dtype=dtype)
        return LaplaceResult(
            covariance=empty_matrix,
            hessian=empty_matrix,
            MAP_estimates=MAP_estimates,
            parameter_index=parameter_index,
        )

    raw_params = [model.covar_module.get_raw_param(k) for k in parameters_with_gradient]
    parameter_index = {name: i for i, name in enumerate(parameters_with_gradient)}

    grads = torch.autograd.grad(loss,
                                raw_params,
                                create_graph=True,)
    n = len(raw_params)
    raw_hessian = torch.zeros((n, n), device=device, dtype=dtype)
    for i, grad_i in enumerate(grads):
        second = torch.autograd.grad(grad_i,
                                     raw_params,
                                     retain_graph=True,
                                     allow_unused=True,)
        raw_first_derivative.append(grad_i)
        if parameter_has_constraint[i]:
            constraint = getattr( model.covar_module,f"raw_{parameters_with_gradient[i]}_constraint")
            CM = ConstraintsModifications(constraint)
            inv_transformation_first_derivative[i], inv_transformation_second_derivative[i] = CM.inverse_derivatives(model.covar_module.get_param(parameters_with_gradient[i]))
        for j, val in enumerate(second):
            if val is not None:
                raw_hessian[i, j] = val

    hessian = torch.zeros((len(parameters_with_gradient), len(parameters_with_gradient)), device=device, dtype=dtype)
    for i in range(len(parameters_with_gradient)):
            for j in range(len(parameters_with_gradient)):
                if i == j: 
                    hessian[i,i] = raw_hessian[i,i]*inv_transformation_first_derivative[i]**2+raw_first_derivative[i]*inv_transformation_second_derivative[i]
                else: 
                     hessian[i,j] = raw_hessian[i,j]*inv_transformation_first_derivative[i]*inv_transformation_first_derivative[j]
    covariance_matrix = torch.linalg.inv(hessian)
    covariance_matrix_out = covariance_matrix.cpu().detach()
    hessian_out = hessian.cpu().detach()   
    return LaplaceResult(covariance = covariance_matrix_out, hessian = hessian_out, parameter_index = parameter_index, MAP_estimates = MAP_estimates)#covariance_matrix, hessian.cpu().detach(), parameters_with_gradient #need to return parameters with gradient so that we know which matrixelement corresponds to which parameter
