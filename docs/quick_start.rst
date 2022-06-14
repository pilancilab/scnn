.. _Quick Start:

Quick Start
===========

Installation
------------

``scnn`` requires ``Python 3.6`` or newer. The package is available through ``pip``,

.. code:: bash

   pip install scnn

Alternatively, you can use the most recent build on GitHub,

.. code:: bash

   git clone https://github.com/pilancilab/scnn
   python -m pip install .


Training Models
---------------

``scnn`` reformulates training a neural network as a convex program.
The easiest way to solve this program is to call :func:`optimize <scnn.optimize.optimize>` on a training set with a few additional configuration parameters:

.. code:: python
   
   from scnn.optimize import optimize
   from scnn.regularizers import NeuronGL1

   model, metrics = optimize(formulation="relu", 
                             max_neurons=500, 
                             X_train=X, 
                             y_train=y, 
                             regularizer=NeuronGL1(0.001),
                             device="cpu") 

   # training accuracy
   train_acc = np.sum(np.sign(model(X_train)) == y_train) / len(y_train)

The ``formulation`` parameter specifies what kind of neural network to train.
We currently support convex reformulations for two-layer models with the following activation functions:

- :code:`"relu"`: the standard rectified linear unit (ReLU),
  
.. math:: \phi(x, w) = \max\{\langle w, x\rangle, 0\} 

- :code:`"gated_relu"`: the Gated ReLU activation, which uses a fixed gate vector :math:`g \in \mathbb{R}^d` when computing the activation,

.. math:: \phi_g(x, w) = \mathbf{1}(\langle g, x\rangle > 0) \langle w, x \rangle

By default, :func:`optimize <scnn.optimize.optimize>` returns a *neuron sparse* solution.
The maximum size of the hidden layer is controlled by ``max_neurons``, while the degree of sparsity is tuned by the regularization strength, ``lam``. 
GPU acceleration is supported and specified using the ``device`` parameter.


The Object-Oriented Interface
-----------------------------

The ``scnn`` module provides an object-oriented interface for greater control over the problem parameters and solver settings.
The following example trains the same ReLU model using the object-oriented interface:

.. code-block:: python

   from scnn import ConvexReLU, RFISTA, Metrics, optimize_model, sample_gate_vectors

   # create convex reformulation
   max_neurons = 500
   G = sample_gate_vectors(np.random.default_rng(123), d, max_neurons)
   model = ConvexReLU(G)
   # specify regularizer and solver
   regularizer = NeuronGL1(lam=0.001)
   solver = AL(model, tol=1e-6)
   # choose metrics to collect during training
   metrics = Metrics(model_loss=True, 
                     train_accuracy=True, 
                     test_accuracy=True, 
                     neuron_sparsity=True) 
   # train model!
   model, metrics = optimize_model(model,
                                   solver,
                                   metrics,
                                   X_train, 
                                   y_train, 
                                   X_test, 
                                   y_test,
                                   regularizer,
                                   device="cpu")

   # training accuracy
   train_acc = np.sum(np.sign(model(X_train)) == y_train) / len(y_train)

Instead of specifying the formulation to solve, we directly instantiate a ``ConvexReLU`` model by passing it a matrix of gate vectors.
The number of gate vectors is analogous to choice of ``max_neurons`` in the :func:`optimize <scnn.optimize.optimize>` function --- see :ref:`Models` for more details.
We also the solver to use (:class:`AL <scnn.solvers.AL>`), a sparsity-inducing regularizer (:class:`NeuronGL1 <scnn.regularizers.NeuronGL1>`) and the metrics to collect during optimization (:class:`Metrics <scnn.metrics.Metrics>`).


Next Steps
----------

See the :ref:`Documentation` for further details on the models, regularizers, and solvers supported by ``scnn``. 
Or, get hands-on experience training neural networks with convex optimization using the :ref:`Examples`.
