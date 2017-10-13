import numpy as np
# from calculate_ors import calculate_dist
from linidx import linidx_take


def calculate_catalogue(dist):
    # print(dist)
    prefactor = np.power(4.0 / (np.pi), 3)
    dist_size = dist.shape[0]
    catalogue = np.zeros((256, dist_size, dist_size))
    catalogue = [i / dist[:, :] for i in range(256)]
    catalogue *= 2
    catalogue *= np.arctan(catalogue)
    catalogue -= np.log(np.square(catalogue) + 1)
    catalogue -= np.square(np.arctan(catalogue))
    # print(catalogue)
    return prefactor * catalogue


def integrand_catalogue(catalogue, height_diff):
    integrand = linidx_take(catalogue, height_diff)
    return integrand


if __name__ == '__main__':

    dist = calculate_dist(5, 5, 11, 11)
    catalogue = calculate_catalogue(dist)
    height_diff = np.random.randint(256, size=(11, 11))
    print(height_diff)
    integrand = integrand_catalogue(catalogue, height_diff)
    print(integrand)
