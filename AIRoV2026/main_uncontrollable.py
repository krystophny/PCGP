import torch
import gpytorch
import Bipendulum as ex
import PCGP.gpytorch.gpytorch_tools as gt
import numpy as np
import math
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
import copy

sigma = 0.03
BC = np.array([[1., 0.], [0., 1.]])
train_x_2 = torch.linspace(0, 6, 10)
train_i_2 = torch.ones_like(train_x_2)*2
train_y_2 = torch.sin(2*train_x_2) + sigma*torch.randn_like(train_x_2)

train_x_0 = torch.Tensor([0., 6.])
train_i_0 = torch.zeros_like(train_x_0)
train_y_0 = torch.Tensor(BC[0])# + sigma*torch.randn_like(train_x_0)

train_x_1 = torch.Tensor([0., 6.])
train_i_1 = torch.ones_like(train_x_1)
train_y_1 = torch.Tensor(BC[1]) #+ sigma*torch.randn_like(train_x_1)

train_x = torch.cat([train_x_0, train_x_1, train_x_2])
train_i = torch.cat([train_i_0, train_i_1, train_i_2])
train_y = torch.cat([train_y_0, train_y_1, train_y_2])
full_train_x = torch.stack([train_x, train_i], dim=-1)

num_tasks = 3

parameters = { "amplitude": [5., True, gpytorch.constraints.Positive()], 
                "lengthscale": [1.9, True, gpytorch.constraints.Positive()],}

noise_tensor = torch.tensor(sigma**2*np.ones(train_y.shape[0]))
noise_tensor[(full_train_x[:,1] != 2)] = 1e-6
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise_tensor)
model = ex.PCGP_Model(full_train_x, train_y, likelihood, parameters, num_tasks = num_tasks)

model.train()
likelihood.train()
N_training = 100
parameters_during_training = {}
for key in parameters:
        parameters_during_training[key] = []
optimizer = torch.optim.Adam(model.named_parameters(), lr=0.1)
marginal_log_likelihood  = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for i in range(N_training):
        optimizer.zero_grad()
        output = model(full_train_x)
        loss = -marginal_log_likelihood(output, train_y)
        for key in parameters:
            for kernel in model.covar_module.kernels:
                if key in kernel.parameters:
                    parameters_during_training[key].append(copy.deepcopy(kernel.get_param(key).detach()))
        if i%100==0:
            print("iteration: ", i, "loss:", loss.item())
        loss.backward(retain_graph = True)
        optimizer.step()

for key in parameters_during_training:
    parameters_during_training[key] =  torch.stack(parameters_during_training[key]).detach().cpu().numpy()


model.eval()
likelihood.eval()
test_x = torch.linspace(0, 6, 100)

test_i = torch.cat([torch.zeros_like(test_x), torch.ones_like(test_x), torch.ones_like(test_x)*2], dim=0)
full_test_x = torch.stack([test_x.repeat(3), test_i], dim=-1)
mean_0, lower_0, upper_0 = gt.predict(model, test_x=torch.stack([test_x, torch.zeros_like(test_x)], dim=-1))
mean_1, lower_1, upper_1 = gt.predict(model, test_x=torch.stack([test_x, torch.ones_like(test_x)], dim=-1))
mean_2, lower_2, upper_2 = gt.predict(model, test_x=torch.stack([test_x, torch.ones_like(test_x)*2], dim=-1))



def analytic_solution(train_x, BC, t1 = 6, wu = 2, g = 1, l = 1, noise = 0):
    tt = train_x[:,0]
    index = train_x[:,1]
    w = math.sqrt(g/l)
    A= 1.
    scaling = np.sin(t1*w)

    theta = np.zeros((tt[index == 0].shape[0], 2))
    for i in range(2):
        t = tt[index == i]
        theta[:,i] = np.sin(w*(t1-t))/scaling*BC[0,i] + np.sin(w*t)/scaling*BC[1,i] + (A/l)/(w**2 - wu**2)*(np.sin(wu*t) - np.sin(wu*t1)/scaling*np.sin(w*t)) 
    theta += noise*np.random.randn(*theta.shape)
    t = tt[index == 2]
    u = A*np.sin(wu*t)
    u += np.random.randn(*u.shape)*noise
    y = np.stack([theta[:,0], theta[:,1],u], axis=-1)
    return y


true_solution = analytic_solution(full_test_x.numpy(), BC, noise = 0)
import os

with torch.no_grad():
    np.savez_compressed(
            os.path.join(os.path.dirname(__file__), "data_uncontrollable.npz"),
            test_x = full_test_x.numpy(),
            test_y = true_solution,
            train_x = full_train_x.numpy(),
            train_y = train_y.numpy(),
            mean = torch.stack([mean_0, mean_1, mean_2], dim = -1).numpy(),
            lower = torch.stack([lower_0, lower_1, lower_2], dim = -1).numpy(),
            upper = torch.stack([upper_0, upper_1, upper_2], dim = -1).numpy()
            )
print("saved")




plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(num_tasks,1, sharex=True, figsize=(8,7))
ax[0].plot(test_x.numpy(), mean_0.detach().numpy(), "b-")
ax[0].fill_between(test_x.numpy(), lower_0.detach().numpy(), upper_0.detach().numpy(), alpha=0.5)
ax[0].plot(test_x.numpy(), true_solution[:,0], "r--")
ax[0].plot(train_x[train_i==0].numpy(), train_y[train_i==0].numpy(), "k*")
ax[1].plot(test_x.numpy(), mean_1.detach().numpy(), "b-")
ax[1].fill_between(test_x.numpy(), lower_1.detach().numpy(), upper_1.detach().numpy(), alpha=0.5)
ax[1].plot(test_x.numpy(), true_solution[:,1], "r--")
ax[1].plot(train_x[train_i==1].numpy(), train_y[train_i==1].numpy(), "k*")
ax[2].plot(test_x.numpy(), mean_2.detach().numpy(), "b-")
ax[2].fill_between(test_x.numpy(), lower_2.detach().numpy(), upper_2.detach().numpy(), alpha=0.5)
ax[2].plot(test_x.numpy(), true_solution[:,2], "r--")
ax[2].plot(train_x[train_i==2].numpy(), train_y[train_i==2].numpy(), "k*")

ax[0].set_ylabel(r"""$\theta_1$""")
ax[1].set_ylabel(r"""$\theta_2$""")
ax[2].set_ylabel(r"""$u$""")
ax[2].set_xlabel(r"""$t$""")
ax[0].legend(["Mean", "Confidence", "True", "Observations"], ncol = 4, loc=(-.07, 1.05),  fontsize = 14)

plt.tight_layout(h_pad = 0.1)
plt.savefig("Bipendulum_uncontrollable.svg")
plt.show()
