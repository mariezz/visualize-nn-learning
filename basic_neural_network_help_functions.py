import numpy as np

def f1(X):  # toy target function
    return np.exp(X/5)

def f2(X):  # toy target function
    return np.sin(X) * (- X * X + 25)

def sigmoid(X):  # sigmoid activation function
    return 1/(1+np.exp(-X))


# Code for reproducible random numbers:

seed = 222337497072607291153223098371913928704  # a random seed
rng = np.random.default_rng(seed) # a random number generator

N_train = 1000  # number of training datapoints
N_test = 100  # number of test datapoints
X_train = (rng.random(N_train) - 0.5) * 10  # 1000 datapoints drawn from interval [-5,5)
X_test = (rng.random(N_test) - 0.5) * 10  # 100 datapoints drawn from interval [-5,5)

def get_train_set(f):  # always returns the same randomly generated dataset
    X_train_copy = np.copy(X_train)
    return (X_train_copy, f(X_train_copy))

def get_test_set(f):  # always returns the same randomly generated dataset
    X_test_copy = np.copy(X_test)
    return (X_test_copy, f(X_test_copy))

def get_new_random_dataset(f, N):  # generates a new random dataset
    X = (np.random.rand(N) - 0.5) * 10  # N datapoints drawn from interval  [-5,5)
    return (X, f(X))

def get_initial_weights(h):  # always returns the same randomly generated initial weihts
    # We want the function to return the same weights each time (for reproducibility and to avoid unpredictable behavior in the notebook)
    # We cannot generate the weights before the call (because we need h), so the function must call the generator
    # To get the same output each time, we need to reset the generator before we use it
    seed_weights = 36255793825986746149655859236796938907
    rng_weights = np.random.default_rng(seed_weights)
    U = rng_weights.normal(0, 0.01, size=(h-1,2))  # small random numbers
    V = rng_weights.normal(0, 0.01, size=(1,h))  # small random numbers
    return (U, V)

def get_new_random_weights(h):  # generates new random weights
    U = np.random.randn(h-1,2) / 100  # small random numbers
    V = np.random.randn(1,h) / 100  # small random numbers
    return (U, V)

                     