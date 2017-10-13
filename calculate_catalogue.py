import numpy as np
# from calculate_ors import calculate_dist
from linidx import linidx_take


def calculate_catalogue(dist):
    """ Calculates a catalogue of integrand values for all possible
     combinations of height differences and distances. By quantizing the
     image as 8-bits, there are only 256 possible height differences,
     which means this is actually feasible.

    Args:
        dist: Matrix with all distances from pixel of interest to the
        square around it. It is used both to know how big the catalogue
        needs to be and for calculating the possible slopes.

    Returns:
        catalogue: a three-dimensional array with integrand values
        for all possible values of slope and height difference
        """
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
    catalogue = prefactor * catalogue
    return catalogue


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
