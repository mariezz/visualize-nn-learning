"""
README!

This program produces dynamic visualizations of the learning process of our basic neural network.
Below you can find explanations about the figure and tips to speed up the execution.

THE FIGURE:
At the top of the figure, you can see the iteration number and the elapsed time.
Plot 1 shows the dataset, this plot is static and doesn't change during or after execution.
Plot 2 shows the learned fonction (which is updated after epoch) together with the target function.
Plot 3 shows the evolution of the training and test errors.
Plot 4 shows the evolution of the norm of the gradient.
Plot 5 shows the values of the angles between consecutive gradients.
Plot 6 shows
    - during execution: a visualization of the 30 last gradient descent steps
    - after execution: a visualization of all gradient descent steps, red points every hundred of iterations (so you can
      better see which arrows correspond to which part of the learning process), and red crosses that indicate where the
      gradient norm and training errors are minimal. Zoom on the plot (using the magnifier at the edge of the figure) to
      have a closer look at the arrows.
Plot 7,8,9 show the evolution of some of the parameter values.

SPEEDING UP THE EXECUTION:
Updating all plot at each iteration takes a lot of time.
To speed up execution, you can try the following:
- During execution: press the right arrow to not plot the next 100 iterations, which speeds up those iterations by a
  factor of 10.
- Before execution: set the variables 'show_grad' and/or 'show_weights' to 'False'. This will speed up execution by not
  showing the plots 4-6 and or 7-9 during execution but only after execution. I encourage you to choose 'show_grad=True'
  and 'show_weights=False', as the gradient plots give more information during execution.

VARIABLES TO PLAY WITH:
You can try different values of the following variables:
- f: the target function
- h: the number of hidden neurons
- learning_rate
- epochs: the number of iterations
- show_grad: a boolean variable that says whether the plots of the gradient should be shown during execution
- show_weights: a boolean variable that says whether the plots of the weights should be shown during execution
- index_lists: a list of list that says which weights should be plotted in the last three plots
Of course, you can also try to change other things in the code.

VALUES YOU SHOULD TRY:
(f, h, learning_rate, epochs) = (f2, 8, 0.1, 1200) and (f1, 3, 1, 1000)
    We've seen their plots already in the notebook, so you know which iterations are interesting and which ones to skip.
(show_grad, show_weights) = (True, False)
    All plots will be shown at the end so you can look at them in detail after the execution.

"""


import numpy as np
import matplotlib.pyplot as plt
import time as time

from basic_neural_network_help_functions import f1, f2
from basic_neural_network_help_functions import get_train_set, get_test_set, get_new_random_dataset
from basic_neural_network_help_functions import get_initial_weights, get_new_random_weights
from basic_neural_network_help_functions import sigmoid


# Compute the network's output for 1 datapoint
def forward_pass(x):
    X[0] = x  # And X[1]=1
    H[:h-1] = sigmoid(U @ X)  # And H[h-1]=1
    O = V @ H
    return O[0]


# Compute the gradient of the loss on 1 datapoint
def back_propagate(e):  # e=o-y is the error (while the loss is half of the square of e)
    grad_V = e * H
    grad_U = np.zeros(U.shape)
    for i in range(h-1):
        grad_U[i,:] = e * V[0,i] * H[i] * (1 - H[i]) * X
    return grad_V, grad_U


# Compute the loss on a given dataset:
def compute_loss(x_values, y_values):
    loss = 0
    for j in range(len(x_values)):
        error = forward_pass(x_values[j]) - y_values[j]
        loss += error*error/2
    loss /= len(x_values)  # divide the loss by the size of the dataset
    return loss


# Compute the loss on the test set:
def compute_test_loss():
    return compute_loss(X_test, Y_test)


# Compute the loss on the training set:
def compute_train_loss():
    return compute_loss(X_train, Y_train)


def plot_dataset():
    ax_dataset.plot(X_train, Y_train, "ro", label="Train set")
    ax_dataset.plot(X_test, Y_test, "bo", label="Test set")
    ax_dataset.set_xlabel("x")
    ax_dataset.set_ylabel("y")
    ax_dataset.legend()


def plot_learned_function_init():
    ax_learned_function.plot(np.arange(-5., 5., 0.1), [f(x) for x in np.arange(-5., 5., 0.1)], "b",
                             label="true function")
    line, = ax_learned_function.plot(np.arange(-5., 5., 0.1), [forward_pass(x) for x in np.arange(-5., 5., 0.1)], "r",
                                     label="learned function")
    ax_learned_function.set_xlabel("x")
    ax_learned_function.set_ylabel("y")
    ax_learned_function.legend()
    return line


def plot_learned_function():
    line1.set_ydata([forward_pass(x) for x in np.arange(-5., 5., 0.1)])


def plot_error_init():
    # ax_error.set_xlim(-3,epochs+3)
    line_test, = ax_error.semilogy(E_test, label="test set error")
    line_train, = ax_error.semilogy(E_train, label="train set error")
    ax_error.set_xlabel("iterations")
    ax_error.set_ylabel("error")
    ax_error.legend()
    return line_test, line_train


def plot_error():  # plot the evolution of the error on the train and test sets
    line2.set_ydata(E_test)
    line3.set_ydata(E_train)
    ax_error.relim()
    ax_error.autoscale_view()


def plot_error_final():
    plot_error()
    # Plot additional information with the error:
    ax_error.plot([epochs], [E_test[-1]], "ro", label=f"last test-e: {'{:.4f}'.format(E_test[-1])}")
    ax_error.plot([np.argmin(E_test)], [min(E_test)], "mx",
                  label=f"min test-e: {'{:.4f}'.format(min(E_test))} (iter {np.argmin(E_test)})")
    ax_error.plot([np.argmin(E_train)], [np.min(E_train)], "rx", label=f"min train-e (iter {np.argmin(E_train)})")
    ax_error.legend()


# Note: the evolution of the error and of the parameters contain epochs+1 values, while the evolution of the gradient's
# norm contains epochs values and there are epochs-1 angles between consecutive gradients.
# The error is computed already before the first iteration, the gradient norm after the first iter, and the angles after
# the second iter.
# So I plot the error starting from iter 0, the norm starting from iter 1, and the angle starting from iter 2.
# In the notebook I kept it simple and didn't play with the iteration numbers so all x-axes start at 0.

def visualize_gradients_init():
    # gradient_norm is only calculated at the end of the first iteration, whereas the error was already computer before
    line_gnorm, = ax_grad_norm.semilogy([i for i in range(1, 1+len(gradient_norm))], gradient_norm, label="norm of the gradient")
    ax_grad_norm.legend()
    ax_grad_norm.set_xlabel("iterations")
    ax_grad_norm.set_ylabel("norm")

    ax_grad_angl.set_ylim(-5, 180 + 5)
    # gradient angles can be computed after the second iteration
    line_gangl, = ax_grad_angl.plot([i for i in range(2, 2+len(gradient_angles))], np.degrees(gradient_angles),
                                    label="angle between consecutive gradients")  # first convert radians to degrees
    ax_grad_angl.legend()
    ax_grad_angl.set_xlabel("iterations")
    ax_grad_angl.set_ylabel("degrees")

    return line_gnorm, line_gangl


def visualize_gradients(points, updates):
    line4.set_ydata(gradient_norm)
    ax_grad_norm.relim()
    ax_grad_norm.autoscale_view()
    line5.set_ydata(np.degrees(gradient_angles))
    ax_grad_angl.relim()
    ax_grad_angl.autoscale_view()

    ax_grad_visu.cla()
    ax_grad_visu.quiver(points[0,:], points[1,:], updates[0,:], updates[1,:], angles="xy", scale_units="xy", scale=1, color="b")
    # ax_grad_visu.margins(0.1)  # margins don't solve the problem when arrows are vertical/horizontal and aligned
    ax_grad_visu.axis("off")
    ax_grad_visu.axis("equal")
    # TODO: solve problem of not fully visible arrows


def visualize_gradients_final():
    visualize_gradients(gradient_2d_points[:, :], gradient_2d_updates[:, :])
    ax_grad_visu.plot(gradient_2d_points[0, ::100], gradient_2d_points[1, ::100], "r.")
    ax_grad_visu.plot([gradient_2d_points[0, minima[0]], gradient_2d_points[0, minima[1]]],  # point mith min grad norm
                      [gradient_2d_points[1, minima[0]], gradient_2d_points[1, minima[1]]], "rx")  # min train error

    # Plot additional information
    ax_grad_norm.semilogy([np.argmin(gradient_norm)+1], [np.min(gradient_norm)], "rx",
                          label=f"minimum (iter {np.argmin(gradient_norm) + 1})")
    ax_grad_norm.legend()


def plot_weights_init():
    lines_u1 = [None for _ in range(len(index_lists[0]))]
    for k in range(len(index_lists[0])):
        (i,j) = index_lists[0][k]
        if i >= h - 1 or j >= 2: continue  # skip if the indexes are out of bounds
        lines_u1[k], = ax_U1.plot(evolution_U[i, j, :], label=f"U{i}{j}")  # plot the evolution of Uij
    ax_U1.set_xlabel("iterations")
    ax_U1.set_ylabel("U")
    ax_U1.legend()

    lines_u2 = [None for _ in range(len(index_lists[1]))]
    for k in range(len(index_lists[1])):
        (i,j) = index_lists[1][k]
        if i >= h - 1 or j >= 2: continue  # skip if the indexes are out of bounds
        lines_u2[k], = ax_U2.plot(evolution_U[i, j, :], label=f"U{i}{j}")  # plot the evolution of Uij
    ax_U2.set_xlabel("iterations")
    ax_U2.set_ylabel("U")
    ax_U2.legend()

    lines_v = [None for _ in range(len(index_lists[2]))]
    for k in range(len(index_lists[2])):
        i = index_lists[2][k]
        if i < 0: continue
        lines_v[k], = ax_V.plot(evolution_V[0, i, :], label=f"V[{i}]")
    ax_V.set_xlabel("iterations")
    ax_V.set_ylabel("V")
    ax_V.legend()

    return lines_u1, lines_u2, lines_v


def plot_weights():
    for k in range(len(index_lists[0])):
        (i,j) = index_lists[0][k]
        if i >= h - 1 or j >= 2: continue  # skip if the indexes are out of bounds
        lines_u1[k].set_ydata(evolution_U[i, j, :])  # plot the evolution of Uij)
    ax_U1.relim()
    ax_U1.autoscale_view()

    for k in range(len(index_lists[1])):
        (i,j) = index_lists[1][k]
        if i >= h - 1 or j >= 2: continue  # skip if the indexes are out of bounds
        lines_u2[k].set_ydata(evolution_U[i, j, :])  # plot the evolution of Uij
    ax_U2.relim()
    ax_U2.autoscale_view()

    for k in range(len(index_lists[2])):
        i = index_lists[2][k]
        if i < 0: continue
        lines_v[k].set_ydata(evolution_V[0, i, :])
    ax_V.relim()
    ax_V.autoscale_view()


def update_figure(current_iter, nb_iter):
    fig.suptitle(f"Iteration {current_iter}/{nb_iter}, elapsed time: {'{:.2f}'.format(time.time()-start)} seconds")
    fig.canvas.draw()
    fig.canvas.flush_events()


def on_key_press(event):
    global next_epoch_to_plot
    if event.key == "right":
        next_epoch_to_plot += 100  # skip 100 iterations


if __name__ == '__main__':
    print("During execution, you can press the right arrow to not plot the next 100 iterations, which speeds up those "
          "iterations by a factor of 10.")

    # README
    # I invite you to play with f2, h, learning_rate, epochs, show_grad (True or False), show_weights (True or false)
    # and the index_lists

    # DEFAULT VALUES: f1, 3, 1, 1000, True, False
    # or f2, 8, 0.1, 1200, True, False
    # + give info about what is interesting (TODO)

    f = f1  # Target function, f1 or f2
    h = 3  # Number of hidden neurons, e.g. 2, 3, 5, 8, ...
    learning_rate = 1  # 1, 0.1, 0.05
    epochs = 1000  # 100, 1000, 10000 ...

    show_grad = True  # show plots of the gradients (plots 4-6)
    show_weights = False  # show plots of the weights (plots 7-9)

    X_train, Y_train = get_train_set(f)  # 1000 datapoints
    X_test, Y_test = get_test_set(f)  # 100 datapoints
    # X_train, Y_train = get_new_random_dataset(f, 1000)
    # X_test, Y_test = get_new_random_dataset(f, 100)

    # Initialize the input and hidden layers:
    X = np.ones(2)
    H = np.ones(h)

    # Initialize the weights to small random numbers:
    U, V = get_initial_weights(h)  # U has size (h-1)x2 and V has size 1xh
    # U, V = get_new_random_weights(h)

    # List that say which elements of U and V will be plotted:
    index_lists = [[(0,0), (1,0), (2,0), (3,0)], [(0,1), (1,1), (2,1), (3,1)], [h-5, h-4, h-3, h-2, h-1]]

    # Keep track of the evolution of the error and of the parameters (so we can plot the evolution later):
    E_test = [None for i in range(epochs + 1)]
    E_test[0] = compute_test_loss()
    E_train = [None for i in range(epochs + 1)]
    E_train[0] = compute_train_loss()
    evolution_V = np.full((1, h, epochs + 1), np.nan)
    evolution_V[:, :, 0] = V
    evolution_U = np.full((h - 1, 2, epochs + 1), np.nan)
    evolution_U[:, :, 0] = U
    evolution_grad_V = np.zeros((1, h, epochs))
    evolution_grad_U = np.zeros((h - 1, 2, epochs))
    gradient_norm = np.full(epochs, np.nan)
    gradient_angles = np.full(epochs - 1, np.nan)  # angles, in radians, between consecutive gradients
    gradient_2d_points = np.zeros((2, epochs + 1))
    gradient_2d_updates = np.zeros((2, epochs + 1))  # only e updates, but array of size e+1 to match the points array
    gradient_2d_angle = 0  # current angle on the 2D plot that visualizes the gradients

    # Make the figure:
    plt.ion()
    fig = plt.figure(figsize=(16, 12))
    ax_dataset = fig.add_subplot(3,3,1)
    ax_learned_function = fig.add_subplot(3,3,2)
    ax_error = fig.add_subplot(3,3,3)
    if show_grad:
        ax_grad_norm = fig.add_subplot(3,3,4)
        ax_grad_angl = fig.add_subplot(3,3,5)
        ax_grad_visu = fig.add_subplot(3,3,6)
    if show_weights:
        ax_U1 = fig.add_subplot(3,3,7)
        ax_U2 = fig.add_subplot(3,3,8)
        ax_V = fig.add_subplot(3,3,9)

    next_epoch_to_plot = 0
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Initialize the plots:
    plot_dataset()
    line1 = plot_learned_function_init()
    line2, line3 = plot_error_init()
    if show_grad: line4, line5 = visualize_gradients_init()
    if show_weights: lines_u1, lines_u2, lines_v = plot_weights_init()

    start = time.time()

    update_figure(0, epochs)
    update_figure(0, epochs)   # one is not enough to stop the screen from being black

    for i in range(epochs):
        # Estimate the gradient using the full dataset:
        grad_V = np.zeros(V.shape)
        grad_U = np.zeros(U.shape)
        for j in range(len(X_train)):
            o = forward_pass(X_train[j])
            e = o - Y_train[j]
            gradients = back_propagate(e)
            grad_V += gradients[0]
            grad_U += gradients[1]

        # Update the weights:
        V -= learning_rate * grad_V / len(X_train)
        U -= learning_rate * grad_U / len(X_train)

        # Store the error on the test and train sets, the weights, and the gradients for later analysis:
        E_test[i + 1] = compute_test_loss()
        E_train[i + 1] = compute_train_loss()
        evolution_V[:, :, i + 1] = V
        evolution_U[:, :, i + 1] = U
        evolution_grad_V[:, :, i] = grad_V / len(X_train)
        evolution_grad_U[:, :, i] = grad_U / len(X_train)

        # Compute and store the norm of the gradient and angles with previous gradient:
        gradient_norm[i] = np.sqrt(
            np.sum(np.square(evolution_grad_V[:, :, i])) + np.sum(np.square(evolution_grad_U[:, :, i])))
        if i > 0:
            dot_product = np.sum(np.multiply(evolution_grad_V[:, :, i - 1], evolution_grad_V[:, :, i])) \
                          + np.sum(np.multiply(evolution_grad_U[:, :, i - 1], evolution_grad_U[:, :, i]))
            cosinus = dot_product / (gradient_norm[i - 1] * gradient_norm[i])
            gradient_angles[i - 1] = np.arccos(cosinus)
            if i % 2 == 0: gradient_2d_angle -= gradient_angles[i - 1]
            if i % 2 == 1: gradient_2d_angle += gradient_angles[i - 1]
        gradient_2d_updates[:, i] = [gradient_norm[i] * np.cos(gradient_2d_angle),
                                     gradient_norm[i] * np.sin(gradient_2d_angle)]
        gradient_2d_points[:, i + 1] = gradient_2d_points[:, i] + gradient_2d_updates[:, i]

        if i == next_epoch_to_plot:
            plot_learned_function()
            plot_error()
            if show_grad: visualize_gradients(gradient_2d_points[:, max(0, i - 30):i+2],
                                              gradient_2d_updates[:, max(0, i - 30):i+2])
            # iter i+1 included, update not yet calculated thus currently zero, so end point is included in the plot
            if show_weights: plot_weights()
            update_figure(i+1, epochs)  # This line takes all the time, even when above 4 are commented
            next_epoch_to_plot += 1
        else:
            assert i < next_epoch_to_plot

    minima = [np.argmin(gradient_norm), np.argmin(E_train)]

    # Finalize the plots:
    plot_learned_function()
    plot_error_final()
    if not show_grad:
        ax_grad_norm = fig.add_subplot(3, 3, 4)
        ax_grad_angl = fig.add_subplot(3, 3, 5)
        ax_grad_visu = fig.add_subplot(3, 3, 6)
        line4, line5 = visualize_gradients_init()
    visualize_gradients_final()
    if not show_weights:
        ax_U1 = fig.add_subplot(3, 3, 7)
        ax_U2 = fig.add_subplot(3, 3, 8)
        ax_V = fig.add_subplot(3, 3, 9)
        lines_u1, lines_u2, lines_v = plot_weights_init()
    plot_weights()

    update_figure(epochs, epochs)

    plt.ioff()
    plt.show()








