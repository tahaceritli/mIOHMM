from src.piomhmm import mHMM
from src.utils import save_pickle
import matplotlib.pyplot as plt
import numpy as np
import torch


def pred(model_name, model, params, b_hat):
    model_mps = model.predict_sequence(params, n_sample=b_hat)

    xhat = np.zeros((n, t))
    xvar = np.zeros((n, t))

    for i in range(n):
        for j in range(t):
            idx = np.where(model_mps[i, j].cpu().numpy() == np.arange(model.k))[0][0]
            if model_name in ["HMM", "mHMM"]:
                xhat[i, j] = params["mu"][idx].cpu().numpy()
            else:
                xhat[i, j] = (
                    params["mu"][idx].cpu().numpy() + b_hat[i].cpu().detach().numpy()
                )
            xvar[i, j] = 2 * np.sqrt(params["var"][idx].cpu().numpy())

    return {"xhat": xhat, "xvar": xvar}


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=2)
device = "cpu"

# DATA GENERATION
n = 200  # number of samples
d = 1  # dimensionality of observations
t = 30  # number of time steps
k = 2  # number of states
K = 2  # number of HMM mixtures

# set parameters
A_1 = torch.tensor([[0.8, 0.2], [0.2, 0.8]])  # transition matrix
A_2 = torch.tensor([[0.2, 0.8], [0.8, 0.2]])  # transition matrix
A = torch.block_diag(*[A_1, A_2])
pi = torch.ones(k * K) / (k * K)  # initial state distribution
mu = torch.tensor([0.0, 2.0, 0.0, 2.0])  # state means
var = torch.tensor([0.1, 0.1, 0.1, 0.1])  # state covariance
b = 1.0  # specify the range of a uniform distribution over personalized state effects, e.g. r_i ~ Unif[-b, b]

# simulate the model
X = torch.zeros((n, t, d))
Z = torch.zeros((n, t), dtype=torch.long)
for i in range(n):
    for j in range(t):
        if j == 0:
            Z[i, j] = torch.multinomial(pi, num_samples=1).byte()
            # D[i, j] = torch.rand(1)
            m_dist = torch.distributions.normal.Normal(
                mu.index_select(0, Z[i, j]), var.index_select(0, Z[i, j])
            )
            X[i, j, :] = m_dist.sample()
        else:
            Z[i, j] = torch.multinomial(A[Z[i, j - 1], :], num_samples=1)
            # D[i, j] = torch.rand(1)
            m_dist = torch.distributions.normal.Normal(
                mu.index_select(0, Z[i, j]), var.index_select(0, Z[i, j])
            )
            X[i, j, :] = m_dist.sample()

# add noise
X_hat = torch.zeros(n, t, d)
l = 1.0  # lengthscale for the SE kernel
s = 0.1  # sigma^2 for the SE kernel
# build covariance matrix
var_x = torch.zeros(t, t)
t_vec = torch.range(0, t)
for j in range(t):
    for jj in range(t):
        r = (t_vec[j] - t_vec[jj]) ** 2
        var_x[j, jj] = s * torch.exp(-r / (2 * l ** 2))
L = torch.cholesky(var_x)
b_stor = torch.zeros(n)
for i in range(n):
    e = torch.randn(t)
    b_stor[i] = 2 * b * torch.rand(1) - b
    X_hat[i, :, :] = (
        torch.einsum("ik,k->i", [L, e])[None, :, None]
        + X[i, :, :]
        + b_stor[i] * torch.ones(1, t, 1)
    )

# plot a number of samples generated
fig, axs = plt.subplots(3, 5, dpi=200)
fig.set_size_inches(12, 5)
for i, ax in enumerate(axs.flatten()):
    ax.plot(X[i, :].numpy(), label="$x_i$")

    ax.plot(X_hat[i, :].numpy(), label="$\hat{x}_i$")

    ax.set_title("Sample " + str(i))
fig.tight_layout()
ax.legend(
    loc="lower center", bbox_to_anchor=(-2.2, -0.95), fancybox=True, shadow=True, ncol=5
)

# FITTING MODELS
print("fitting standard HMM...")
hmm = mHMM(
    X_hat,
    k=k,
    K=1,
    full_cov=False,
    priorV=False,
    io=False,
    personalized=False,
    personalized_io=False,
    state_io=False,
    device=device,
    eps=1e-18,
)
hmm_params, _, ll_hmm = hmm.learn_model(num_iter=10000, intermediate_save=False)

print("fitting standard mHMM...")
mhmm = mHMM(
    X_hat,
    k=k,
    K=K,
    full_cov=False,
    priorV=False,
    io=False,
    personalized=False,
    personalized_io=False,
    state_io=False,
    device=device,
    eps=1e-18,
)
mhmm_params, _, ll_mhmm = mhmm.learn_model(num_iter=10000, intermediate_save=False)

print("fitting personalized HMM...")
phmm = mHMM(
    X_hat,
    k=k,
    K=1,
    full_cov=False,
    priorV=False,
    io=False,
    personalized=True,
    personalized_io=False,
    state_io=False,
    device=device,
    eps=1e-18,
)
phmm_params, _, _, elbo_phmm, b_hat_phmm, _ = phmm.learn_model(
    num_iter=10000, intermediate_save=False
)

print("fitting personalized mHMM...")
mphmm = mHMM(
    X_hat,
    k=2,
    K=2,
    full_cov=False,
    priorV=False,
    io=False,
    personalized=True,
    personalized_io=False,
    state_io=False,
    device=device,
    eps=1e-18,
)
mphmm_params, _, _, elbo_mphmm, b_hat_mphmm, _ = mphmm.learn_model(
    num_iter=10000, intermediate_save=False
)

# RESULTS
# save outputs
outputs = {
    "HMM": {"model": hmm, "params": hmm_params, "b_hat": None},
    "mHMM": {"model": mhmm, "params": mhmm_params, "b_hat": None},
    "PHMM": {"model": phmm, "params": phmm_params, "b_hat": b_hat_phmm},
    "mPHMM": {"model": mphmm, "params": mphmm_params, "b_hat": b_hat_mphmm},
}
preds = {}
for model_name in outputs:
    model = outputs[model_name]["model"]
    params = outputs[model_name]["params"]
    b_hat = outputs[model_name]["b_hat"]
    preds[model_name] = pred(model_name, model, params, b_hat)
save_pickle({"outputs": outputs, "preds": preds}, "outputs/synthetic_all.pkl")

# figure
fig, axs = plt.subplots(3, 4, dpi=200)
fig.set_size_inches(12, 8)
for j in range(3):
    for i in range(4):
        ax = axs[j][i]
        ax.plot(X_hat[i, :].numpy(), "k:", label="$\hat{x}_i$")

        if j == 0:
            xhat = preds["HMM"]["xhat"]
            xvar = preds["HMM"]["xvar"]
            ax.plot(xhat[i, :], label="HMM $\mu_k \pm 2\sigma_{k,i}$")
            ax.fill_between(
                np.arange(t),
                xhat[i, :] - xvar[i, :],
                xhat[i, :] + xvar[i, :],
                alpha=0.5,
            )

            xhat = preds["PHMM"]["xhat"]
            xvar = preds["PHMM"]["xvar"]
            ax.plot(xhat[i, :], label="PHMM $(\mu_k + r^{(i)}) \pm 2\sigma_{k,i}$")
            ax.fill_between(
                np.arange(t),
                xhat[i, :] - xvar[i, :],
                xhat[i, :] + xvar[i, :],
                alpha=0.5,
            )
        elif j == 1:
            xhat = preds["mHMM"]["xhat"]
            xvar = preds["mHMM"]["xvar"]
            ax.plot(xhat[i, :], label="mHMM $\mu_k \pm 2\sigma_{k,i}$")
            ax.fill_between(
                np.arange(t),
                xhat[i, :] - xvar[i, :],
                xhat[i, :] + xvar[i, :],
                alpha=0.5,
            )

            xhat = preds["mPHMM"]["xhat"]
            xvar = preds["mPHMM"]["xvar"]
            ax.plot(xhat[i, :], label="mPHMM $(\mu_k + r_i) \pm 2\sigma_{k,i}$")
            ax.fill_between(
                np.arange(t),
                xhat[i, :] - xvar[i, :],
                xhat[i, :] + xvar[i, :],
                alpha=0.5,
            )
        else:
            xhat = preds["PHMM"]["xhat"]
            xvar = preds["PHMM"]["xvar"]
            ax.plot(xhat[i, :], label="PHMM $(\mu_k + r_i) \pm 2\sigma_{k,i}$")
            ax.fill_between(
                np.arange(t),
                xhat[i, :] - xvar[i, :],
                xhat[i, :] + xvar[i, :],
                alpha=0.5,
            )

            xhat = preds["mPHMM"]["xhat"]
            xvar = preds["mPHMM"]["xvar"]
            ax.plot(xhat[i, :], label="mPHMM $(\mu_k + r_i) \pm 2\sigma_{k,i}$")
            ax.fill_between(
                np.arange(t),
                xhat[i, :] - xvar[i, :],
                xhat[i, :] + xvar[i, :],
                alpha=0.5,
            )

        ax.set_xlabel("Time")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_title("Sample " + str(i))
        ax.set_ylim([-4, 4])
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-2.7, -0.42),
        fancybox=True,
        shadow=True,
        ncol=4,
    )

fig.tight_layout()
fig.subplots_adjust(hspace=0.7)
fig.savefig(
    "outputs/synthetic_x_hats.png",
    dpi=400,
    facecolor="w",
    edgecolor="w",
    orientation="portrait",
    bbox_inches="tight",
    pad_inches=0,
    metadata={"Creator": None, "Producer": None, "CreationDate": None},
)
