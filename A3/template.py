from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def filter1(img, filter, off):
    pass


def filter2(img, filter, off, edge):
    pass

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


if __name__ == "__main__":

    # read img
    img = io.imread("lena.jpg")

    # convert to numpy array
    img = np.array(img)

    # convert to grayscale
    img = rgb2gray(img)

    fm = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    imgOut = filter2(img, fm , 0, 'min')

    # plot img
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img, cmap = cm.Greys_r)
    # plot imgOut
    plt.figure(1)
    plt.subplot(212)
    plt.imshow(imgOut, cmap = cm.Greys_r)

    plt.show()

