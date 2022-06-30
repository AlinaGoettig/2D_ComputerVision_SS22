import cv2
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import math

# mode 0 vertikal spiegeln, mode 1 horizontal
def mirror_img(t_img, mode=0):
    if mode == 0:
        inv_img = np.zeros(t_img.shape, int)
        col = t_img.shape[1]
        for x in range(col):
            inv_img[:, x] = t_img[:,col - x - 1]
        return inv_img
    elif mode == 1:
        inv_img = np.zeros(t_img.shape, int)
        rows = t_img.shape[0]
        for y in range(rows):
            inv_img[y] = t_img[rows - y - 1]
        return inv_img
    else:
        return t_img

# transform image to grayscale
def rgb2gray(img):
    # Source BildverarbeitungBuch: 12.2.1 Umwandlung in Grauwertbilder
    imgCopy = np.copy(img)
    gray = ((imgCopy[:, :, 0] * 0.299) + (imgCopy[:, :, 1] * 0.587) + (imgCopy[:, :, 2] * 0.114)).astype(int)
    return gray

# create Histogramm. bins are optionally implemented
def computeHisto(t_img, bin=None):
    if bin is not None:
        K = 256
        B = np.math.ceil(K / bin)
        t_histo = np.zeros(B, int)
        for pixel in t_img.flatten():
            i = np.math.floor(pixel / bin)
            t_histo[i] += 1
        return t_histo
    else:
        t_histo = np.zeros(256, int)
        for pixel in t_img.flatten():
            t_histo[pixel] += 1
        return t_histo

# print your image
def printImage(t_img, grayscale=True):
    if grayscale:
        io.imshow(t_img, cmap=plt.cm.gray)
        io.show()
    else:
        io.imshow(t_img)
        io.show()

# print your Historgram
def printHisto(t_histo, t_img=""):
    # plot image
    if t_img is not "":
        printImage(t_img, grayscale=True)
    # plot histogram
    n = t_histo.size
    x = range(n)
    width = 1
    plt.bar(x, t_histo, width, color="blue")
    plt.xlim([0, n - 1])
    plt.show()

def apply_lut(img, lut):
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            img[i, j] = lut[pixel]
    return img

def cum_histo(histo, normalized=True):
    kum_his = np.zeros(histo.shape, int)
    for i, entry in enumerate(histo):
        if i == 0:
            kum_his[i] = histo[i]
        else:
            kum_his[i] = kum_his[i-1] + histo[i]
    if normalized:
        return kum_his / kum_his[kum_his.size - 1]
    return kum_his

def match_Histo(img_histo, ref_histo):
    max = 256
    lut = np.arange(0, max, 1)
    for a in lut:
        j = max - 1
        while True:
            lut[a] = j
            j = j - 1
            if ((j >= 0) and (img_histo[a] >= ref_histo[j])):
                break
    return lut

def filter(in_image, filter, off, edge='none'):
    copy_img = in_image.copy()
    width = copy_img.shape[0]
    height = copy_img.shape[1]
 
    for v in range(off, height - off):
        for u in range(off, width - off):
            # fill the pixel vector P for filter position u,v)
            sum = 0
            for j in range(0, filter.shape[0]):
                for i in range(0, filter.shape[1]):
                    c = filter[i][j]
                    a_idx = u + (i - int(filter.shape[0] / 2))
                    b_idx = v + (j - int(filter.shape[1] / 2))

                    # handle padding
                    if a_idx >= 0 and a_idx < width and b_idx >= 0 and b_idx < height:
                        p = copy_img[a_idx][b_idx]
                    else:
                        if edge.__eq__('min'):
                            p = 0
                        elif edge.__eq__('max'):
                            p = 255
                        elif edge.__eq__('continue'):
                            if a_idx < 0:
                                a_idx = 0
                            elif a_idx >= width:
                                a_idx = width - 1
                            if b_idx < 0:
                                b_idx = 0
                            elif b_idx >= height:
                                b_idx = height - 1
                            p = copy_img[a_idx][b_idx]
                        else:
                            p = copy_img[a_idx][b_idx]

                    sum = sum + c * p
            q = int(sum)
            in_image[u][v] = q
    return in_image

def medianFilter(in_image, filtersize, offset):
    copy_img = in_image.copy()
    width = copy_img.shape[0]
    height = copy_img.shape[1]
    
    # vector to hold pixels from 3x3 neighborhood
    P = np.zeros(filtersize * filtersize)
    
    for v in range(offset, height - offset):
        for u in range(offset, width - offset):
            # fill the pixel vector P for filter position u,v)
            k = 0
            for j in range(-int(filtersize / 2), int(filtersize/ 2)+1):
                for i in range(-int(filtersize / 2), int(filtersize/ 2)+1):
                    P[k] = copy_img[u+i][v+j]
                    k += 1
            # sort the pixel vector and take center element
            P = np.sort(P, kind='heapsort')
            in_image[u][v] = P[int(filtersize**2 / 2)]
    return in_image

sobel_1 = np.array([[1,2,1]]) / 4
sobel_2 = np.array([[-1,0,1]]) / 2

def horizontal(img):
    img_copy = img.copy()
    return filter(img_copy, np.transpose(sobel_1) @ sobel_2, 1, "continue")


def vertical(img):
    img_copy = img.copy()
    return filter(img_copy, np.transpose(sobel_2) @ sobel_1, 1, "continue")


def gradient(img):
    return np.sqrt(horizontal(img)**2 + vertical(img)**2)

def get_r_idx(r, max, step):
    result = r / max * step
    return result if round(result) < step else result - 1

def linearHT(im_edge, angle_steps=100, radius_steps=100):
    output = np.zeros((radius_steps, angle_steps))
    r_max = math.hypot(len(im_edge[0]), len(im_edge))
    for line_idx, line in enumerate(im_edge):
        for pxl_idx, pxl in enumerate(line):
            if pxl != 255:
                for theta_idx in range(0, angle_steps):
                    theta = 1.0 * theta_idx * math.pi / angle_steps
                    r = abs(pxl_idx * math.cos(theta) + line_idx * math.sin(theta))
                    r_idx = get_r_idx(r, r_max, radius_steps)
                    output[round(r_idx)][theta_idx] += 1
    return output

def linearHT_th(img, threshold):
    img_copy = img.copy()
    out = []
    img_copy = img_copy.flatten()
    img_copy.sort()
    min = img_copy[-1] * (100-threshold)/100
    for indR, pxl in enumerate(img):
        for indTheta, theta in enumerate(pxl):
            if theta > min:
                out.append((indR, indTheta))
    return out

def printEdge(edges, width, img):
    #test = edges[0]
    #x = [float(tmp) for tmp in range(0, width)]
    #y = -(x * math.cos(test[1]) - test[0]) / math.sin(test[1])
    #plt.plot(x, y, 'r')
    #plt.show()
    plt.scatter(*zip(*edges))
    plt.show()

def sequential_labelling(img, print=True):
    m = 2
    c = list()

    # part 1
    for line_idx in range(0, len(img)):
        for pxl_idx in range(0, len(img[0])):
            if img[line_idx][pxl_idx] == 1:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if (line_idx + i) < 0 or (pxl_idx + j) < 0:
                            continue
                        if line_idx + i > len(img) - 1 or pxl_idx + j > len(img[0]) - 1:
                            continue
                        n = img[line_idx + i][pxl_idx + j]
                        if n > 1 and n != img[line_idx][pxl_idx]:
                            if img[line_idx][pxl_idx] == 1:
                                img[line_idx][pxl_idx] = n
                            else:
                                s = set()
                                s.add(img[line_idx][pxl_idx])
                                s.add(n)
                                c.append(s)
                if img[line_idx][pxl_idx] == 1:
                    img[line_idx][pxl_idx] = m
                    m += 1
    if print is True:
        printImage(img)

    # part 2
    for i in range(len(c)):
        for j in range(len(c)):
            if c[i].__eq__(c[j]):
                continue
            s_tmp = c[j]
            for value in s_tmp:
                if value in c[i]:
                    c[i] = c[i].union(c[j])
                    c[j] = set()
                    break

    # part 3
    for line_idx in range(len(img)):
        for pxl_idx in range(len(img[0])):
            if img[line_idx][pxl_idx] > 1:
                for s in c:
                    if img[line_idx][pxl_idx] in s:
                        img[line_idx][pxl_idx] = min(s)
    return img


def to_bin(img):
    result = img.copy()
    max_val = np.max(img)
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if img[i][j] >= max_val / 2:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result


def dilate(in_image, filter, iter_num):
    new_img = in_image.copy()

    for x in range(0, iter_num):
        t_img = np.zeros(new_img.shape)

        jc = int((filter.shape[0] - 1) / 2)
        ic = int((filter.shape[1] - 1) / 2)

        for j in range(0, filter.shape[0]):
            for i in range(0, filter.shape[1]):
                if filter[j][i] == 1:
                    tmp = np.roll(new_img, j - jc, axis=0)

                    # fix np.roll rollover (2,3,4 -> left shift 4,2,3 we want 0, 2, 3)
                    if j - jc < 0:
                        tmp[tmp.shape[0] + (j - jc):] = 0
                    elif j - jc > 0:
                        tmp[:(j - jc)] = 0

                    tmp = np.roll(tmp, i - ic, axis=1)
                    if i - ic < 0:
                        tmp[:, tmp.shape[1] + (i - ic):] = 0
                    elif i - ic > 0:
                        tmp[:, :(i - ic)] = 0

                    t_img = np.fmax(tmp, t_img)
        new_img = t_img

    return new_img


def erode(in_image, filter, iter_num, debug=False):
    inverted_img = 255 - in_image
    if debug:
        printImage(inverted_img)
    t_img = dilate(inverted_img, filter, iter_num)
    if debug:
        printImage(t_img)
    result = 255 - t_img
    if debug:
        printImage(result)
    return result


def open(in_img, filter, iter):
    tmp = in_img.copy()
    tmp = erode(tmp, filter, iter)
    tmp = dilate(tmp, filter, iter)
    return tmp


def close(in_img, filter, iter):
    tmp = in_img.copy()
    tmp = dilate(tmp, filter, iter)
    tmp = erode(tmp, filter, iter)
    return tmp
