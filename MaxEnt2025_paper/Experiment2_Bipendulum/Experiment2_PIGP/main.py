import sys
import os
import torch
import gpytorch
import Experiment2_PIGP as ex
import PCGP.gpytorch.gpytorch_tools as gt
import numpy as np
torch.set_default_dtype(torch.float64)
import jax.numpy as jnp
import jax.random as random
import Experiment2_PIGP_numpyro as nmcmc

if len(sys.argv) != 5:
    print("Usage: python script.py <npoints> <sigma> <end> <run>")
    sys.exit(1)

npoints = int(sys.argv[1])
sigma = float(sys.argv[2])
end = int(sys.argv[3])
run = int(sys.argv[4])
torch.manual_seed(100+run**2)
data_path = os.path.join(
    os.path.dirname(__file__),
    "../input_data",
    "Ex2_n%i_sigma%.2f_end%i.npz"%(npoints, sigma, end)
)
data = np.load(data_path)

train_x = torch.tensor(data["train_x"])
test_x = torch.tensor(data["test_x"])
train_y = torch.tensor(data["train_y_PIGP"])
test_y = torch.tensor(data["test_y_PIGP"])


do_MCMC = False
num_samples = 15000
l1_true = 1.
l2_true = 2.
g_true = 9.81

num_tasks = train_y.shape[1]

parameters = {"l1": [1. +torch.randn(1)*.5, True, gpytorch.constraints.Interval(0.5, 4.)],
              "l2": [2. + torch.randn(1)*.5, True, gpytorch.constraints.Interval(0.5, 4.)],
            "g": [9.81, False, gpytorch.constraints.Positive()],
            "A": [10., False, gpytorch.constraints.Positive()], 
            "l": [1.4, False, False]}#gpytorch.constraints.Interval(train_x[1]-train_x[0], train_x[-1]-train_x[0])],}
priors = {"l1": gpytorch.priors.UniformPrior(0.5, 4.),
          "l2": gpytorch.priors.UniformPrior(0.5, 4.)}
fixed_params = {}
for key in parameters:
    if not parameters[key][1]:
        fixed_params[key] = parameters[key][0]

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, has_global_noise = False, noise_constraint=gpytorch.constraints.Positive())                                                                                                                               
model = ex.PCGP_Model(train_x, train_y, likelihood, parameters, num_tasks = num_tasks, priors=priors)
if sigma == 0:
    sigma = 1e-3
gt.fix_task_noises(torch.tensor([sigma, sigma, sigma, sigma])**2, model)

model.train()
likelihood.train()
N_training = 500
learning_rate = 0.05
param_training, laplace_covariance, hessian, used_parameters, loss_landscape, test_loss = gt.train(model, 
                                                                                                                         likelihood, 
                                                                                                                         parameters, 
                                                                                                                         train_x, 
                                                                                                                         train_y, 
                                                                                                                         num_tasks,
                                                                                                                         test_x = test_x, 
                                                                                                                         test_y = test_y,                                                                                                                           
                                                                                                                         noise_constraints=None, 
                                                                                                                         training_iter= N_training)
#param_training, laplace_covariance, hessian, used_parameters, loss_landscape = gt.train(model, likelihood, parameters, train_x, train_y, num_tasks, noise_constraints=None, training_iter = N_training)


samples = {"l1":torch.zeros(1), "l2":torch.zeros(1)}
if do_MCMC:
    X = jnp.array(train_x)
    Y = jnp.concatenate([jnp.array(train_y[:,i]) for i in range(num_tasks)], axis = 0)
    rng_key = random.PRNGKey(run)
    mcmc_model = nmcmc.model
    init_value = {key: param_training[key][-1] for key in used_parameters}
    init_value["l1"] = l1_true
    init_value["l2"] = l2_true
    samples = nmcmc.run_inference(mcmc_model, rng_key, X, Y, num_samples = num_samples, sigma = sigma, 
                                  init_values = init_value,
                                  fixed_params=fixed_params)
    


print("l1 = ", model.covar_module.get_param("l1"), )
print("l2 = ", model.covar_module.get_param("l2"), )
print(laplace_covariance)

model.eval()
likelihood.eval()
mean, lower, upper = gt.predict(model, likelihood, test_x=test_x)

parameters_during_training = torch.stack([torch.tensor(param_training[key]) for key in used_parameters], dim = 0)

output_path = os.path.join(
    os.path.dirname(__file__),
    "../output_data",
    "fixedAl/Ex2_PIGP_n%i_sigma%.2f_end%i_%i.npz"%(npoints, sigma, end, run)
)
with torch.no_grad():   
    np.savez_compressed(
            output_path,
            mean=np.array(mean),
            test_x = test_x.numpy(),
            train_x = train_x.numpy(),
            train_y = train_y.numpy(),
            test_y = test_y.numpy(), 
            lower=np.array(lower),
            upper = np.array(upper),
            learned_parameters = np.squeeze(np.array([param_training[key][-1] for key in used_parameters])),
            true_parameters = np.array([l1_true, l2_true]),
            laplace_covariance = np.array(laplace_covariance),
            hessian = np.array(hessian),
            samples_l1 = np.array(samples["l1"]),
            samples_l2 = np.array(samples["l2"]),
            loss = np.array(loss_landscape),
            test_loss = np.array(test_loss),
            parameters_during_training = parameters_during_training.numpy()
        )
print("saved")
#plt.show()
