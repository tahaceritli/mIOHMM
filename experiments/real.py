from src.piomhmm import mHMM
import numpy as np
import random
import torch
from src.utils import save_pickle, load_pickle

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=2)


def preprocess(x, d):
    # don't include samples which only have one measurement, i.e. aren't time series
    remove_idx = np.where(np.sum(~np.isnan(x[:, :, 0]), axis=1) == 1)
    x = np.delete(x, remove_idx, 0)
    d = np.delete(d, remove_idx, 0)
    # set any LEDD values greater than 5000 to 620 and rescale
    d[d > 5000] = 620
    d[np.isnan(d)] = 0
    d = d / np.max(d)
    # get time and observation masks
    N, T, D, = x.shape
    time_mask = np.ones((N, T))
    for i in range(N):
        ind = np.where(~np.isnan(x[i, :, 0]))[0][-1] + 1
        time_mask[i, ind:] = 0
    missing_mask = (~np.isnan(x[:, :, 0])).astype(float)
    x[np.isnan(x)] = 0
    # convert everything to tensors
    X = torch.Tensor(x).float()
    D = torch.Tensor(d).float()
    TM = torch.Tensor(time_mask).float()
    OM = torch.Tensor(missing_mask).bool()
    return X, D, TM, OM


# data
data = load_pickle("processed/data_for_PIOHMM.pkl")
X_train, D_train, TM_train, OM_train = preprocess(data["x_train"], data["train_med"])
X_test, D_test, TM_test, OM_test = preprocess(data["x_test"], data["test_med"])

# experiment setting
device = "cpu"
k = 8  # number of hidden states
num_iter_train = 10000
num_iter_test = 5000

for K in [1, 2, 3, 4, 5]:

    # mIOHMM
    print("fitting mIOHMM...", flush=True)
    model = mHMM(
        X_train,
        ins=D_train,
        k=k,
        K=K,
        TM=TM_train,
        OM=OM_train,
        full_cov=False,
        io=True,
        personalized=False,
        personalized_io=False,
        state_io=True,
        UT=True,
        device=device,
        eps=1e-18,
    )
    params_hat, e_out, ll = model.learn_model(
        num_iter=num_iter_train, intermediate_save=False
    )
    training_pX = model.calc_pX(params_hat)

    mIOHMM_model = {
        "params": params_hat,
        "e_out": e_out,
        "ll": ll,
        "training_pX": training_pX,
    }

    print("learning vi params ...\n", flush=True)
    model.change_data(
        X_test, ins=D_test, TM=TM_test, OM=OM_test, reset_VI=True, params=params_hat
    )
    params_hat, e_out_test, ll_test = model.learn_vi_params(
        params_hat, num_iter=num_iter_test
    )
    test_pX = model.calc_pX(params_hat)

    mIOHMM_model["params_test"] = params_hat
    mIOHMM_model["e_out_test"] = e_out_test
    mIOHMM_model["ll_test"] = ll_test
    mIOHMM_model["test_pX"] = test_pX

    save_pickle(
        mIOHMM_model, "models/mIOHMM_" + str(K) + ".pkl",
    )
