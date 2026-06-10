import os
import numpy as np
import torch
import gpytorch
import PCGP.gpytorch.gpytorch_tools as gt
import Experiment1_Helmholtz as ex
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_r, n_p, n_bc, ntest = 10, 15, 30, 21
data_path = os.path.join(
    os.path.dirname(__file__),
    "input_data",
    "Ex1_nr%i_np%i_nbc%i_ntest%i.npz" % (n_r, n_p, n_bc, ntest)
)
data = np.load(data_path)   
train_x = torch.tensor(data["train_x"]).to(device)
test_x = torch.tensor(data["test_x"]).to(device)
train_y = torch.tensor(data["train_y"]).to(device)
f_true = torch.tensor(data["f_true"]).to(device)
xx = test_x[:,0].reshape((ntest, ntest))
yy = test_x[:,1].reshape((ntest, ntest))


parameters = {"A":[1., True, gpytorch.constraints.Positive()], 
            "l": [0.5, True, gpytorch.constraints.Interval(0.1, 1.)],}
num_tasks = 3

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, has_global_noise = False, noise_constraint=gpytorch.constraints.Interval(1e-10, 5e-5 ))                                                                                                                               
model = ex.PCGP_Model(train_x, train_y, likelihood, parameters, num_tasks = num_tasks, priors=None)
model.to(device)
likelihood.to(device)

model.train()
likelihood.train()
param_training, inverse_hessian, hessian, used_parameters, loss_landscape, _ = gt.train(model, likelihood, 
                                                               parameters, train_x, train_y, 
                                                               num_tasks, training_iter = 500)
parameters_during_training = torch.stack([torch.tensor(param_training[key]) for key in used_parameters], dim = 0)
model.eval()
likelihood.eval()
test_x.requires_grad_(True)

with gpytorch.settings.observation_nan_policy("mask"):
    pred = model(test_x)
print("finished predicting")    


def compute_laplacian(output, inputs, shape):
    grad = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output, device = device),
        create_graph=True,
        retain_graph=True  # Needed if called multiple times
    )[0] 

    laplacian_terms = []
    for i in range(inputs.shape[1]):
        grad_i = grad[:, i]
        grad2_i = torch.autograd.grad(
            outputs=grad_i,
            inputs=inputs,
            grad_outputs=torch.ones_like(grad_i, device = device),
            create_graph=True,
            retain_graph=True
        )[0][:, i]
        laplacian_terms.append(grad2_i)

    laplacian = sum(laplacian_terms)
    return laplacian.reshape(*shape)


lap_A1 = compute_laplacian(pred.mean[:,0], test_x, (ntest, ntest))
lap_A2 = compute_laplacian(pred.mean[:,1], test_x, (ntest, ntest))
print("calculated_laplacian")

A1 = pred.mean[:, 0].detach().reshape(ntest, ntest)
A2 = pred.mean[:, 1].detach().reshape(ntest, ntest) 
f_predicted = pred.mean[:, 2].detach().reshape(ntest, ntest) 


with torch.no_grad():
    error = torch.stack([lap_A1 + 30*torch.cos(2*xx)*A1 +50*(yy+1)**3*A2 - f_true, lap_A2 + 10*A1 + A2, f_true - f_predicted], dim = 0)

output_path = os.path.join(
    os.path.dirname(__file__),
    "output_data",
    "Ex1_nr%i_np%i_nbc%i_ntest%i.npz" % (n_r, n_p, n_bc, ntest)
)
with torch.no_grad():
    np.savez_compressed(
            output_path,
            loss = loss_landscape,
            parameters_during_training = parameters_during_training.cpu().numpy(),
            test_x = torch.stack([xx, yy], dim = 0).cpu().numpy(),
            test_y = torch.stack([A1, A2, f_true], dim = 0).cpu().numpy(),
            error = error.cpu().numpy(),
            train_x = train_x.cpu().numpy(),
            train_y = train_y.cpu().numpy(),
            A1 = A1.detach().cpu().numpy(),
            A2 = A2.detach().cpu().numpy(),
            f_pred = f_predicted.detach().cpu().numpy(),
            f_true = f_true.detach().cpu().numpy(),
            )
print("saved")