
import torch
import gpytorch
from .constraint_handling import ConstraintsModifications
from dataclasses import dataclass
import copy
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@dataclass#parameters_during_training, inverse_hessian, hessian, order_of_parameters_with_gradients, loss_landscape, test_loss
class train_output:
    parameters_during_training:dict
    covariance_matrix: torch.tensor
    hessian:torch.tensor
    order_of_parameters_in_covariance:list
    loss_landscape:list
    test_loss:float



def train(model, likelihood, parameters, train_x, train_y, num_tasks, test_x=None, test_y=None,  noise_constraints = None, training_iter=50, laplace = False):
    with gpytorch.settings.observation_nan_policy("mask"):  
        params = []
        for name, param in model.named_parameters():
            params.append(param)
        parameters_during_training = {}
        for key in parameters:
             parameters_during_training[key] = []
        #parameters_during_training["noise"] = torch.zeros((num_tasks, training_iter), device = device )   
        loss_landscape = []
        optimizer = torch.optim.Adam(params, lr=0.1)
        marginal_log_likelihood  = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -marginal_log_likelihood(output, train_y)*train_y.flatten().shape[0] #############################
                
                for key in parameters:
                    parameters_during_training[key].append(copy.deepcopy(model.covar_module.get_param(key).detach()))
                #for t in range(num_tasks):
                #    parameters_during_training["noise"][t, i] = likelihood.task_noises[t]    
                loss_landscape.append(loss.detach())
                #if noise_constraints and likelihood.task_noises.requires_grad:
                #    noise_constraints.enforce_constraints(likelihood.raw_task_noises)
                if i%100==0 and not device=="cuda":
                    print("iteration: ", i, "loss:", loss.item())
                loss.backward(retain_graph = True)
                optimizer.step()
        
        model.train()
        likelihood.train()        
        optimizer.zero_grad()
        output = model(train_x)  
        #print("output", output, train_x.shape)           
        final_loss = -marginal_log_likelihood(output, train_y)*train_y.flatten().shape[0]###########################################
        test_loss = None
        if test_x is not None and test_y is not None:
            model.eval()
            likelihood.eval()
            test_output = model(test_x)
            test_loss = -marginal_log_likelihood(test_output, test_y).detach()#############################
        inverse_hessian, hessian, order_of_parameters_with_gradients = None, None, None
        if laplace:
            inverse_hessian, hessian, order_of_parameters_with_gradients = calculate_laplace_approx_matrix(parameters, model, final_loss)
        loss_landscape = torch.stack(loss_landscape).detach().cpu().numpy() 
        for key in parameters_during_training:
            if key != "noise":
                parameters_during_training[key] =  torch.stack(parameters_during_training[key]).detach().cpu().numpy()
    return train_output(parameters_during_training=parameters_during_training,
                        covariance_matrix=inverse_hessian,
                        hessian=hessian,
                        order_of_parameters_in_covariance=order_of_parameters_with_gradients,
                        loss_landscape=loss_landscape,
                        test_loss=test_loss)
    #return parameters_during_training, inverse_hessian, hessian, order_of_parameters_with_gradients, loss_landscape, test_loss     


def predict(model, likelihood=None, test_x = torch.linspace(0, 1, 51)):
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.observation_nan_policy("mask"):
        if likelihood:
            predictions = likelihood(model(test_x))
        else:
            predictions = model(test_x)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
    return mean, lower, upper


def fix_task_noises(task_noises, model):
    if torch.is_tensor(task_noises):
        fixed_task_noises = task_noises
    else:
        fixed_task_noises = torch.tensor(task_noises, device=device)  
    fixed_raw_task_noises = model.likelihood.raw_task_noises_constraint.inverse_transform(fixed_task_noises)
    model.likelihood.raw_task_noises.requires_grad = False
    with torch.no_grad():
        model.likelihood.raw_task_noises.copy_(fixed_raw_task_noises)
        
def calculate_laplace_approx_matrix(parameters, model, loss):
    parameters_with_gradient = []
    for key in parameters:
        if model.covar_module.get_param(key).requires_grad:
            parameters_with_gradient.append(key)
    raw_hessian = torch.zeros((len(parameters_with_gradient), len(parameters_with_gradient)), device=device)
    raw_first_derivative = []
    inv_transformation_first_derivative = torch.ones(len(parameters_with_gradient), device=device)
    inv_transformation_second_derivative = torch.zeros(len(parameters_with_gradient), device=device)
    for i in range(len(parameters_with_gradient)):
        key = parameters_with_gradient[i]
        raw_first_derivative.append(torch.autograd.grad(loss, model.covar_module.get_raw_param(key), retain_graph = True, create_graph=True)[0])
        raw_hessian[i,i] = torch.autograd.grad(raw_first_derivative[i], model.covar_module.get_raw_param(key), retain_graph = True)[0]
        if parameters[key][2]:
            CM = ConstraintsModifications(eval("model.covar_module.raw_"+f"{key}_constraint"))
            inv_transformation_first_derivative[i], inv_transformation_second_derivative[i] = CM.inverse_derivatives(model.covar_module.get_param(key))
    for i in range(len(parameters_with_gradient)):
        key_i = parameters_with_gradient[i]
        for j in range(i+1, len(parameters_with_gradient)):
            key_j = parameters_with_gradient[j]
            mixed_derivative = torch.autograd.grad(torch.autograd.grad(loss, model.covar_module.get_raw_param(key_i), retain_graph = True, create_graph=True)[0]
                                                       , model.covar_module.get_raw_param(key_j), retain_graph = True)[0]
            raw_hessian[i,j] = mixed_derivative
            raw_hessian[j,i] = mixed_derivative
    hessian = torch.zeros((len(parameters_with_gradient), len(parameters_with_gradient)), device=device)  
    for i in range(len(parameters_with_gradient)):
            for j in range(len(parameters_with_gradient)):
                if i == j: 
                    hessian[i,i] = raw_hessian[i,i]*inv_transformation_first_derivative[i]**2+raw_first_derivative[i]*inv_transformation_second_derivative[i]
                else: 
                     hessian[i,j] = raw_hessian[i,j]*inv_transformation_first_derivative[i]*inv_transformation_first_derivative[j]
    covariance_matrix = torch.inverse(hessian).cpu().detach()    
    return covariance_matrix, hessian.cpu().detach(), parameters_with_gradient #need to return parameters with gradient so that we know which matrixelement corresponds to which parameter

