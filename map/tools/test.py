import cv2
import numpy as np


def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def visualize_heatmap(image, mask):
    '''
    Save the heatmap of ones
    '''
    masks = norm_image(mask).astype(np.uint8)
    # mask->heatmap
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))    # same shape

    # merge heatmap to original image
    cam = 0.4*heatmap + 0.6*np.float32(image)
    return cam
