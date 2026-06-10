import os
import gpytorch
import torch
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax
import numpyro.distributions as dist
import Experiment3_LumpedHeat as ex
import Experiment3_LumpedHeat_numpyro as nmcmc
import PCGP.gpytorch.gpytorch_tools as gt
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_jax = jax.devices()[0]
torch.manual_seed(2)


sigma = 0.05
data_path = os.path.join(
    os.path.dirname(__file__),
    "input_data",
    "Ex3_sigma%.2f.npz" %(sigma))

data = np.load(data_path)
train_x = torch.tensor(data["train_x"]).to(device)
test_x = torch.tensor(data["test_x"]).to(device)
train_y = torch.tensor(data["train_y"]).to(device)
test_y = torch.tensor(data["test_y"]).to(device)
R_true = torch.tensor(data["R_true"]).to(device)

window_size = 15
test_window = window_size*8 

parameters = parameters = {"R": [1., True, False],
                           "A": [2., False,  gpytorch.constraints.Interval(0.1, 15)],
                           "l": [1., False, gpytorch.constraints.Interval(0.1, 15)],}
fixed_params = {}
for key in parameters:
    if not parameters[key][1]:
        fixed_params[key] = parameters[key][0]

priors = None

priors_mcmc = None
num_tasks = 2

set_init_val = True
set_priors = False
do_MCMC = True
num_samples = 5000

number_of_models = train_x.shape[0]//window_size


parameter_evolution = {}
parameter_evolution_uncertainty= {}

save_mean = np.zeros((number_of_models, test_window, num_tasks))
save_upper = np.zeros((number_of_models, test_window, num_tasks))
save_lower = np.zeros((number_of_models, test_window, num_tasks))

for key in parameters:
    parameter_evolution[key] = []
    parameter_evolution_uncertainty[key] = []


samples_to_save = np.zeros((number_of_models, num_samples))




class SoftLaplacePrior(gpytorch.priors.Prior):
    def __init__(self, loc, scale, validate_args=None):
        self.loc = torch.as_tensor(loc)
        self.scale = torch.as_tensor(scale)
        batch_shape = self.loc.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return -torch.log(2 * self.scale) - torch.nn.functional.softplus(torch.abs(z))





for i in range(number_of_models):
    train_x_fraction = train_x[i*window_size:(i+1)*window_size].to(device)
    train_y_fraction = train_y[i*window_size:(i+1)*window_size,:].to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, has_global_noise = False, noise_constraint=gpytorch.constraints.Interval(1e-5, 0.05**2))
    model = ex.PCGP_Model(train_x_fraction, train_y_fraction, likelihood,  parameters, num_tasks = num_tasks, priors = priors)
    model = model.to(device)
    likelihood = likelihood.to(device)
    fixed_task_noises = torch.tensor([sigma, sigma], device=device)**2
    gt.fix_task_noises(fixed_task_noises, model)

    model.train()
    likelihood.train()
    param_training, inverse_hessian, hessian, used_parameters, loss_landscape, _ = gt.train(model, likelihood, 
                                                               parameters, train_x_fraction, train_y_fraction, 
                                                               num_tasks, training_iter = 500)
  
    for j in range(len(used_parameters)):
        key = used_parameters[j]
        parameter_evolution[key].append(param_training[key][-1])
        parameter_evolution_uncertainty[key].append(torch.sqrt(inverse_hessian[j,j])) 
    model.eval()
    likelihood.eval()
    test_x_fraction = test_x[i*test_window:(i+1)*test_window].to(device)

    mean, lower, upper = gt.predict(model, likelihood, test_x=test_x_fraction)
    save_mean[i] = mean.cpu()
    save_upper[i] = upper.cpu()
    save_lower[i] = lower.cpu()

    if do_MCMC:
        X = jax.device_put(jnp.array(train_x_fraction.cpu().numpy()), device=device_jax)
        Y0 = jax.device_put(jnp.array(train_y_fraction[:,0].cpu().numpy()), device=device_jax)
        Y1 = jax.device_put(jnp.array(train_y_fraction[:,1].cpu().numpy()), device=device_jax)
        Y = jnp.concatenate([Y0, Y1])
        rng_key = random.PRNGKey(0)
        mcmc_model = nmcmc.model
        print([param_training[key][-1] for key in used_parameters])
        samples = nmcmc.run_inference(mcmc_model, rng_key, X, Y, num_samples = num_samples, sigma = sigma, 
                                      init_values = {key: param_training[key][-1] for key in used_parameters},
                                      priors = priors_mcmc, fixed_params=fixed_params)
        MCMC_numpyro = jnp.squeeze(samples["R"])
        samples_to_save[i] = np.array(MCMC_numpyro)

    if set_init_val:
        for key in parameters:
            if parameters[key][1] == True:
                parameters[key][0] = param_training[key][-1,0]

    if set_priors:
        priors = {}
        priors_mcmc = {}
        for key in parameters:
            if key in used_parameters:# and key != "A" and key != "l":
                var = jnp.var(samples[key])
                print(jnp.sqrt(var), "std mcmc")
                mu = jnp.mean(samples[key])
                print("std laplace:", torch.sqrt(inverse_hessian[used_parameters.index(key),used_parameters.index(key)]))
                priors[key] =  SoftLaplacePrior(param_training[key][-1], #SoftLaplacePrior
                                    torch.sqrt(inverse_hessian[used_parameters.index(key),used_parameters.index(key)]))    
                priors_mcmc[key] = dist.SoftLaplace(mu, jnp.sqrt(var))#param_training[key][-1], jnp.array(inverse_hessian[used_parameters.index(key),used_parameters.index(key)]) ) 
            else:    
                priors_mcmc[key] = dist.Normal(0, 1)

output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data",
    "Ex3_no_priors.npz"
)
with torch.no_grad():
        np.savez_compressed(
            output_path,
            laplace_mean=np.array(parameter_evolution["R"]),
            laplace_uncertainty=np.array(parameter_evolution_uncertainty["R"]),
            samples = samples_to_save,
            mean = save_mean,
            lower = save_lower,
            upper = save_upper,
            train_x = train_x.cpu().numpy(),
            test_x = test_x.cpu().numpy(),
            train_y = train_y.cpu().numpy(),
            test_y = test_y.cpu().numpy(),
            R_true = R_true.cpu().numpy(),
        )

print("finished")


