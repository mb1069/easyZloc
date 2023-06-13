import jax.numpy as jnp
from dphtools.utils import fft_pad, slice_maker


def remove_bg(data: jnp.ndarray, multiplier: float) -> jnp.ndarray:
    """Remove background from data.

    Utility that measures mode of data and subtracts a multiplier of it

    Example
    -------
    >>> data = jnp.array([1, 1, 1, 0, 2])
    >>> remove_bg(data, 1)
    array([ 0.,  0.,  0., -1.,  1.])
    """
    # TODO: should add bit for floats, that will find the mode using the hist
    # function bincounts with num bins specified
    # if jnp.issubdtype(data.dtype, jnp.integer):
    #     if jnp.any(data < 0):
    #         raise ValueError("Negative values unsupported")
    #     else:
    #         
    # else:
    #     raise ValueError("Floating point numbers unsupported")
    mode = jnp.bincount(data.ravel(), length=2048).argmax()
    return data - multiplier * mode.astype(float)


def psqrt(data):
    """Take the positive square root, negative values will be set to zero.

    Example
    -------
    >>> psqrt((-4, 4))
    array([0., 2.])
    """
    return jnp.sqrt(jnp.fmax(0, data))


def center_data(data):
    """Center data on its maximum.

    Parameters
    ----------
    data : ndarray
        Array of data points

    Returns
    -------
    centered_data : ndarray same shape as data
        data with max value at the central location of the array

    Example
    -------
    >>> data = np.array(
    ...     (
    ...         (1, 0, 0),
    ...         (0, 0, 0),
    ...         (0, 0, 0),
    ...     )
    ... )
    >>> center_data(data)
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])


    >>> data = np.array(
    ...     (
    ...         (2, 1, 0, 0),
    ...         (1, 0, 0, 0),
    ...         (0, 0, 0, 1),
    ...     )
    ... )
    >>> center_data(data)
    array([[0, 1, 0, 0],
           [0, 0, 2, 1],
           [0, 0, 1, 0]])
    """
    # copy data
    centered_data = data.copy()
    # extract shape and max location
    data_shape = data.shape
    max_loc = jnp.unravel_index(data.argmax(), data_shape)
    # iterate through dimensions and roll data to the right place
    for i, (x0, nx) in enumerate(zip(max_loc, data_shape)):
        centered_data = jnp.roll(centered_data, nx // 2 - x0, i)
    return centered_data


def prep_data_for_PR(data, xysize=None, multiplier=1.05):
    """Prepare data for phase retrieval.

    Will pad or crop to xysize and remove mode times multiplier and clip at zero

    Parameters
    ----------
    data : ndarray
        The PSF data to prepare for phase retrieval
    xysize : int
        Size to pad or crop `data` to along the y, x dimensions
    multiplier : float
        The amount to by which to multiply the mode before subtracting

    Returns
    -------
    prepped_data : ndarray
        The data that has been prepped for phase retrieval.
    """
    # pull shape
    nz, ny, nx = data.shape

    # remove background
    data_without_bg = remove_bg(data, multiplier)

    # figure out padding or cropping
    if xysize is None:
        xysize = max(ny, nx)

    if xysize == ny == nx:
        pad_data = data_without_bg
    elif xysize >= max(ny, nx):
        # pad data out to the proper size, pad with zeros
        pad_data = fft_pad(data_without_bg, (nz, xysize, xysize), mode="constant")
    else:
        # if need to crop, crop and center and return
        my_slice = slice_maker(((ny + 1) // 2, (nx + 1) // 2), xysize)
        pad_data = center_data(data_without_bg)[(Ellipsis,) + my_slice]

    # return centered data
    return jnp.fmax(0, pad_data)
