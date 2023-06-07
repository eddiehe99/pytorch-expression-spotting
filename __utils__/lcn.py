import numpy as np
from scipy.ndimage import convolve
from einops import rearrange, repeat

"""
Repository: https://github.com/nattochaduke/LocalContrastNormalization_Pytorch
"""


def lcn(image, kernel_size=3, sigma=1, mode="constant", cval=0.0):
    """
    :param kernel_size: int, kernel(window) size for local region of image.
    :param sigma: float, the standard deviation of the Gaussian kernel for weight window.
    :param mode: {'reflect', 'constant', 'nearest', 'mirror', 'warp'}, optional
                    determines how the array borders are handled. The meanings are listed in
                    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.convolve.html
                    default is 'constant' as 0, different from the scipy default.
    """

    def gaussian_kernel(kernel_size, sigma):
        """
        :return: kernel_size * kernel_size shape Gaussian kernel with standard deviation sigma
        """
        side = int((kernel_size - 1) // 2)
        x, y = np.mgrid[-side : side + 1, -side : side + 1]
        g = np.exp(
            -(x**2 / (sigma**2 * float(side)) + y**2 / (sigma**2 * float(side)))
        )
        return g / np.sum(g)

    kernel = gaussian_kernel(kernel_size, sigma)

    if len(image.shape) == 3:
        pass
    else:
        # add a dim if the image is in gray scale
        image = repeat(image, "h w -> h w c", c=1)

    # fit for the channel arrangment of the origin repository
    image = rearrange(image, "h w c -> c h w")
    C, H, W = image.shape

    image_v = image - np.sum(
        np.array(
            [convolve(image[c], weights=kernel, mode=mode, cval=cval) for c in range(C)]
        ),
        axis=0,
    )
    image_sigma_square_stack = np.array(
        [
            convolve(np.square(image[c]), weights=kernel, mode=mode, cval=cval)
            for c in range(C)
        ]
    )
    image_sigma = np.sqrt(np.sum(image_sigma_square_stack, axis=0))
    c = np.mean(image_sigma)

    # prevent divide by zero encountered in divide
    eps = np.finfo(image_sigma.dtype).eps
    image_sigma = np.maximum(image_sigma, eps)
    c = np.maximum(c, eps)

    image_v_divided_by_c = image_v / c
    image_v_divided_by_image_sigma = image_v / image_sigma
    image_y = np.maximum(image_v_divided_by_c, image_v_divided_by_image_sigma)
    image_y = rearrange(image_y, "c h w -> h w c")
    return image_y
