import torch
from .constraint_handling import ConstraintsModifications
from dataclasses import dataclass
torch.set_default_dtype(torch.float64)

@dataclass
class LaplaceResult:
    covariance: torch.Tensor
    hessian: torch.Tensor 
    MAP_estimates: torch.Tensor
    parameter_index: dict
    

def laplace_approx(parameters, model, loss):
    """
    Calculates the hessian of the -marginal log likelihood loss.
    Returns a LaplaceResult dataclass containing the hessian, the covariance matrix used in the Laplace Approximation, that is the square root of the inverse of the hessian, the MAP estimates aka the mean in the Laplace Approximation, and a dictionary parameter_index containing the used parameter names as keys and the corresponding indices as values to guarantee the right order when accessing hessian, covariance, and MAP_estimates.
    
    :param parameters: dict
        dictionary of parameters with parameter name strings as keys and list containing [initial value(float), requires_grad(bool), constraints(False or gpytorch.constraints object)
    :param model: Gpytorch model in training mode
    :param loss: calculated loss (implicitly including graph used to calculate it to access gradients)
    """
    device = loss.device
    parameters_with_gradient, parameter_has_constraint = [], []
    for key in parameters:
        if model.covar_module.get_param(key).requires_grad:
            parameters_with_gradient.append(key)
            parameter_has_constraint.append(parameters[key][2])
       
    raw_hessian = torch.zeros((len(parameters_with_gradient), len(parameters_with_gradient)), device=device)
    raw_first_derivative = []
    inv_transformation_first_derivative = torch.ones(len(parameters_with_gradient), device=device)
    inv_transformation_second_derivative = torch.zeros(len(parameters_with_gradient), device=device)
    parameter_index = {}

    MAP_estimates = [model.covar_module.get_param(k).detach() for k in parameters_with_gradient]

    raw_params = [model.covar_module.get_raw_param(k) for k in parameters_with_gradient]
    parameter_index = {name: i for i, name in enumerate(parameters_with_gradient)}

    grads = torch.autograd.grad(loss,
                                raw_params,
                                create_graph=True,)
    n = len(raw_params)
    raw_hessian = torch.zeros((n, n), device=device)
    for i, grad_i in enumerate(grads):
        second = torch.autograd.grad(grad_i,
                                     raw_params,
                                     retain_graph=True,)
        raw_first_derivative.append(grad_i)
        if parameter_has_constraint[i]:
            constraint = getattr( model.covar_module,f"raw_{parameters_with_gradient[i]}_constraint")
            CM = ConstraintsModifications(constraint)
            inv_transformation_first_derivative[i], inv_transformation_second_derivative[i] = CM.inverse_derivatives(model.covar_module.get_param(parameters_with_gradient[i]))
        for j, val in enumerate(second):
            raw_hessian[i, j] = val

    hessian = torch.zeros((len(parameters_with_gradient), len(parameters_with_gradient)), device=device)  
    for i in range(len(parameters_with_gradient)):
            for j in range(len(parameters_with_gradient)):
                if i == j: 
                    hessian[i,i] = raw_hessian[i,i]*inv_transformation_first_derivative[i]**2+raw_first_derivative[i]*inv_transformation_second_derivative[i]
                else: 
                     hessian[i,j] = raw_hessian[i,j]*inv_transformation_first_derivative[i]*inv_transformation_first_derivative[j]
    covariance_matrix = torch.linalg.inv(hessian)

    covariance_matrix_out = torch.sqrt(covariance_matrix.cpu().detach())
    hessian_out = hessian.cpu().detach()   
    return LaplaceResult(covariance = covariance_matrix_out, hessian = hessian_out, parameter_index = parameter_index, MAP_estimates = MAP_estimates)#covariance_matrix, hessian.cpu().detach(), parameters_with_gradient #need to return parameters with gradient so that we know which matrixelement corresponds to which parameter

