from skimage import io
from skimage.color import rgb2gray
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def read_image():
    """Opens a GUI interface, reads an image and converts it to RGB.
    Returns a multi-dimensional numpy ndarray."""
    from tkinter.filedialog import askopenfilename
    filename = askopenfilename()
    image = io.imread(filename)
    image = rgb2gray(image)
    return image


def inv_image(x, y, image):
    """ Simply returns an inverted point of the image.
        y and x are switched because reasons.

    Args:
        x, y: two integer values with horizontal and vertical
        coordinates of the point to be retrieved.
        image: a two-dimensional numpy ndarray with the image data

    Returns:
        a numerical value with the inverse of the desired pixel
        """
    return(-1.0 * image[y][x])


def calculate_height_diff(image, slice_size):
    """ Generates a two-dimensional ndarray with the difference between
    a central point and the rest of the array. Also, gets rid of any
    negative values, because ORS doesn't like those.

    Args:
        image: a two-dimensional numpy ndarray with the image data
        slice_size: a parameter indicating how big the matrix is.
        One could theoretically derive it from the matrix size, but
        it's easier to just use it.

    Returns:
        It uses call-by-value and changes the image arg, so it doesn't
        need to return anything.
        """
    np.add(image, inv_image(slice_size, slice_size, image), out=image,
           casting="unsafe")
    image[image < 0] = 0


def calculate_dist(i, j, max_x, max_y):
    """ Generates a two-dimensional ndarray with the euclidean distances
    between matrix elements and an arbitrary element given by i and j.

    Args:
        i,j: determines the point from which euclidean distances will
        be calculated in the matrix.
        max_x, max_y: total matrix size in both dimensions

    Returns:
        dist: the two-dimensional matrix with distances.
        """
    y, x = np.ogrid[-1 * j:max_y - j, -1 * i:max_x - i]
    dist = np.sqrt(x**2 + y**2)
    return dist


def integrand(slope, prefactor):
    """ Calculates the function to be integrated for calculation of the
    ORS. We are using the function suggested by Earl and Metzler in
    their technical paper about it, but most quadratic functions on
    the slope should be decent. Our function is:
    f(x) = 2 arctan(x) - log(x^2 + 1) - arctan^2(x)

    Args:
        slope: a matrix with slope (height difference divided by
        distance) values. Only positive values are allowed.
        prefactor: we could calculate it here, but why would you do
        that?

    Returns:
        integrand: a two-dimensional matrix with the values of the
        integrand for all the slope values received
        """
    slope *= 2
    slope *= np.arctan(slope)
    slope -= np.log(np.square(slope) + 1)
    slope -= np.square(np.arctan(slope))
    return prefactor * slope


def calculate_ors(image):
    """ Here's the meat of the code. We calculate ORS values for each
    pixel in the image by generating a slice_size sized mask around
    each pixel. It allows us to calculate the distance matrix a single
    time and deal with smaller matrices than if we used the whole
    image, making the code significantly faster.

    Args:
        image: a two-dimensional numpy ndarray with the image data,
        received from the user

    Returns:
        ors: a two-dimensional matrix with ORS values for each pixel
        in the image. Note that, due to the slicing procedure, this is
        smaller than the original image; slice_size pixels on each
        border are lost.
        """

    # slice_size defines the matrix size to be used in the rest of the
    # code. It should be bigger than the expected features in the image.

    slice_size = 30
    max_y, max_x = image.shape

    # generate the distance matrix. It will be the same one for all
    # pixels thanks to the whole slicing idea.

    dist = calculate_dist(slice_size, slice_size, 2 * slice_size + 1,
                          2 * slice_size + 1)

    # initialise the return matrix. it's smaller than the image!

    ors = np.zeros((max_y - 2 * slice_size, max_x - 2 * slice_size))
    prefactor = np.power(4.0 / (np.pi), 3)

    # we will end our for loop short because not all pixels in the
    # image will be analysed (we lose the borders)

    for i, j in product(range(max_x - 2 * slice_size),
                        range(max_y - 2 * slice_size)):

        # this is the magic bit where we take a small chunk of the
        # image around the pixel of interest.

        height_diff = np.copy(image[j:j + 2 * slice_size + 1,
                              i:i + 2 * slice_size + 1])

        # this will be passed to the function where height diffs are
        # calculated.

        calculate_height_diff(height_diff, slice_size)
        slope = np.divide(height_diff, dist)

        # we calculate all integrands, take their square root and sum
        # (i.e. "integrate") over them, ignoring the eventual NaNs

        local_ors = integrand(slope, prefactor)
        local_ors = np.sqrt(local_ors)
        ors[j][i] = np.nansum(local_ors)
    return ors


if __name__ == "__main__":

    """ Main code: gets an image from the user, calculates ORS on it,
    plots everything in the end. Simple and straightforward.
    """

    image = read_image()
    ors = calculate_ors(image)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.imshow(ors, cmap=plt.cm.gray)
    ax1.axis('off')
    plt.show()
