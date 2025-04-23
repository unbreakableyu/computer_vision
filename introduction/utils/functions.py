import numpy as np
from typing import List, Tuple
import cv2
import os

t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    # raise NotImplementedError
    for img, name in zip(images, names):
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    # raise NotImplementedError
    for img, name in zip(images, filenames):
        cv2.imwrite(name, img)


def scale_down(image: np.array) -> np.array:
    """Returns an image half the size of the original.

    Args:
        image: A numpy array with an opencv image

    Returns:
        A numpy array with an opencv image half the size of the original image
    """
    # raise NotImplementedError
    return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))


def separate_channels(colored_image: np.array) -> t_image_triplet:
    """Takes an BGR color image and splits it three images.

    Args:
        colored_image: an numpy array sized [HxWxC] where the channels are in BGR (Blue, Green, Red) order

    Returns:
        A tuple with three BGR images the first one containing only the Blue channel active, the second one only the
        green, and the third one only the red.
    """
    # raise NotImplementedError
    b, g, r = cv2.split(colored_image)
    zero = np.zeros_like(b)

    blue_img = cv2.merge([b, zero, zero])
    green_img = cv2.merge([zero, g, zero])
    red_img = cv2.merge([zero, zero, r])

    return blue_img, green_img, red_img