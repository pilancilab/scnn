import numpy as np
import torch


from scnn.private.utils.data import gen_classification_data

from scnn.optimize import optimize
from scnn.regularizers import NeuronGL1

# Generate realizable synthetic classification problem (ie. Figure 1)
n_train = 1000
n_test = 1000
d = 25
hidden_units = 100
kappa = 1000  # condition number

(X_train, y_train), (X_test, y_test) = gen_classification_data(123, n_train, n_test, d, hidden_units, kappa)

def accuracy(logits, y):
    return np.sum((np.sign(logits) == y)) / len(y)

# cast data and create loader
tX_train, ty_train, tX_test, ty_test = [torch.tensor(z, dtype=torch.float) for z in [X_train, y_train, X_test, y_test]]

# model parameters
lam = 0.001
huber_delta = 1.0 # None for square loss

# number of activation patterns to use.
max_neurons = 1000

cvx_model, metrics = optimize("gated_relu",
                          max_neurons,
                          X_train,
                          y_train,
                          X_test,
                          y_test,
                          loss_type="huber",
                          huber_delta=huber_delta,
                          regularizer=NeuronGL1(lam),
                          verbose=True,
                          device="cpu")

# Acc After Training
print("\n \n")
print("Post-Training Test Accuracy:", accuracy(cvx_model(X_test), y_test))
print(f"Hidden Layer Size: {cvx_model.parameters[0].shape[0]}")