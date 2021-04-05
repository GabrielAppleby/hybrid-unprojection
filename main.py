from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.utils import shuffle


def squared_pairwise_distance(x):
    x_left = tf.expand_dims(x, 0)
    x_right = tf.expand_dims(x, 1)
    squared_difference = tf.square(x_left - x_right)

    return tf.reduce_sum(squared_difference, axis=-1)


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def get_data():
    x, y = datasets.load_iris(return_X_y=True)
    x, y = shuffle(x, y, random_state=42)
    x = x.astype('float32')
    y = y.astype('float32')
    return x, y


def compute_perplexity(distribution):
    distribution += 1e-8
    entropy = -tf.reduce_sum(distribution * log2(distribution), 1)
    perplexity = 2. ** entropy
    return perplexity


def estimate_sigmas(distances,
                    tolerance=1e-4,
                    max_iter=50000,
                    lower_bound=np.float32(1e-20),
                    upper_bound=np.float32(999999999.0),
                    target_perplexity=40.0):
    n = distances.shape[0]

    lower_bound_arr = tf.fill((n, 1), lower_bound)
    upper_bound_arr = tf.fill((n, 1), upper_bound)

    for i in range(max_iter):
        current_sigmas = tf.reshape(((lower_bound_arr + upper_bound_arr) / 2.), (-1, 1))
        current_sigmas_squared_times_two = 2. * tf.square(current_sigmas)
        distances_over_sigmas_squared_times_two = distances / current_sigmas_squared_times_two

        current_conditional_prob_of_neighbors = tf.nn.softmax(
            distances_over_sigmas_squared_times_two, axis=1)
        current_conditional_prob_of_neighbors = tf.linalg.set_diag(
            current_conditional_prob_of_neighbors,
            tf.zeros(n, dtype=tf.float32))

        current_perplexity = compute_perplexity(current_conditional_prob_of_neighbors)

        upper_mask = current_perplexity > target_perplexity
        upper_indices = tf.reshape(tf.where(upper_mask), (-1, 1))
        lower_mask = current_perplexity < target_perplexity
        lower_indices = tf.reshape(tf.where(lower_mask), (-1, 1))

        done_mask = tf.abs(current_perplexity - target_perplexity) <= tolerance
        upper_bound_arr = tf.tensor_scatter_nd_update(upper_bound_arr, upper_indices,
                                                      current_sigmas[upper_mask])
        lower_bound_arr = tf.tensor_scatter_nd_update(lower_bound_arr, lower_indices,
                                                      current_sigmas[lower_mask])

        if tf.math.reduce_all(done_mask):
            print("done")
            return current_sigmas
    raise TimeoutError("Max iterations for sigma binary search exceeded.")


def estimate_high_d_conditional_of_neighbors(x):
    distances = -squared_pairwise_distance(x)
    sigmas = estimate_sigmas(distances)
    sigmas_squared_times_two = 2. * tf.square(sigmas)
    distances_over_sigmas_squared_times_two = distances / sigmas_squared_times_two
    conditional_of_neighbors = tf.nn.softmax(distances_over_sigmas_squared_times_two, 1)
    conditional_of_neighbors = tf.linalg.set_diag(
        conditional_of_neighbors,
        tf.zeros(conditional_of_neighbors.shape[0], dtype=tf.float32))

    return conditional_of_neighbors


def high_d_conditional_to_joint(conditional):
    return conditional + tf.transpose(conditional) / (2. * conditional.shape[0])


def estimate_low_d_joint_of_neighbors(x):
    distances = -squared_pairwise_distance(x)
    inv_distances = tf.math.pow(1. - distances, -1)
    inv_distances = tf.linalg.set_diag(inv_distances, tf.zeros(inv_distances.shape[0]))
    joint_of_neighbors = inv_distances / tf.reduce_sum(inv_distances)
    return joint_of_neighbors


def load_data() -> Tuple[np.array, np.array]:
    """
    Load the data from disk.
    :return: The features, projections, and labels of the data.
    """
    data = np.load("mnist_tsne.npz")
    return data['features'], data['labels']


def main():
    x, y = load_data()
    x = x[0:500].astype(np.float32)
    y = y[0:500].astype(np.float32)

    n, d = x.shape
    high_d_conditional_of_neighbors = estimate_high_d_conditional_of_neighbors(x)
    high_d_joint_of_neighbors = high_d_conditional_to_joint(high_d_conditional_of_neighbors)

    optimizer = tf.keras.optimizers.Adam()
    kld_loss_fn = tf.keras.losses.KLDivergence()
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    low_d_representation = tf.Variable(
        tf.initializers.RandomNormal(stddev=.001)(shape=[n, 2], dtype=tf.float32))
    #
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(d, activation=tf.nn.sigmoid))

    @tf.function
    def train():
        with tf.GradientTape(persistent=True) as tape:
            low_d_joint_of_neighbors = estimate_low_d_joint_of_neighbors(low_d_representation)
            reconstruction = model(low_d_joint_of_neighbors)
            tsne_loss = kld_loss_fn(high_d_joint_of_neighbors, low_d_joint_of_neighbors)
            reconstruction_loss = mse_loss_fn(x, reconstruction)

        var_list = [low_d_representation] + model.trainable_weights
        grads = tape.gradient([tsne_loss, reconstruction_loss], var_list)
        tf.print(tape.gradient([tsne_loss, reconstruction_loss], model.trainable_weights))
        tf.print(tape.gradient([tsne_loss], model.trainable_weights))
        tf.print(tape.gradient([reconstruction], model.trainable_weights))
        optimizer.apply_gradients(zip(grads, var_list))
        return low_d_representation, reconstruction

    for epoch in range(10000):
        low_d, reconstruction = train()
        exit()
        if epoch % 1000 == 0:
            print(epoch)
            show_scatters(low_d, y, "tsne_scatter" + str(epoch))
    show_images(reconstruction[0:15].numpy(), "reconstructions")
    show_images(x[0:15], "originals")


def show_images(images: np.array, name: str) -> None:
    """
    Plots an MNIST image.
    :param images: An MNIST image.
    :param name: The name to save it under.
    :return: None, the images are saved as a side effect.
    """
    plt.gray()
    fig = plt.figure(figsize=(16, 7))
    for i in range(0, 15):
        ax = fig.add_subplot(3, 5, i + 1)
        ax.matshow(images[i].reshape((28, 28)).astype(float))
    plt.savefig(name)
    plt.clf()


def show_scatters(projections: np.array, labels, name: str) -> None:
    """
    Plots a projection scatter plot.
    :param projections: The projections to plot.
    :param labels: The labels to color by.
    :param name: The name to save it under.
    :return: None, the images are saved as a side effect.
    """
    plt.scatter(x=projections[:, 0], y=projections[:, 1], c=labels)
    plt.savefig(name)
    plt.clf()


if __name__ == '__main__':
    main()
