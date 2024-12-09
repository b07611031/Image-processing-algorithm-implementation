import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time


def save_output(img, img_path, suffix='', split=False):
    img[img > 255] = 255
    img[img < 0] = 0
    os.makedirs('./output', exist_ok=True)
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + \
        '_' + suffix + '.' + file_name[-1]
    cv2.imwrite(save_path, img)
    if split:
        if len(suffix) == 3:
            cv2.imwrite(save_path.replace(suffix, suffix[0]), img[:, :, 2])
            cv2.imwrite(save_path.replace(suffix, suffix[1]), img[:, :, 1])
            cv2.imwrite(save_path.replace(suffix, suffix[2]), img[:, :, 0])
    return save_path


# Luma Grayscale
def gray_luma(img, img_path='Output.jpg', suffix=''):
    os.makedirs('./output', exist_ok=True)
    gray = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + suffix + '_gray.' + file_name[-1]
    cv2.imwrite(save_path, gray)
    return save_path


def bgr2rgb(img, img_path, save=False):
    rgb_path = save_output(img, img_path, 'RGB', split=True)
    if save:
        return rgb_path
    rgb_img = img[:, :, [2, 1, 0]]
    return rgb_img, rgb_path


def rgb2cmy(img, img_path):
    output_img = np.zeros(img.shape)
    output_img[:, :, 0] = 255 - img[:, :, 0]
    output_img[:, :, 1] = 255 - img[:, :, 1]
    output_img[:, :, 2] = 255 - img[:, :, 2]
    output_path = save_output(
        output_img[:, :, [2, 1, 0]], img_path, 'CMY', split=True)
    return output_path


def pix_hsi(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*255
    v = mx*255
    return h/360*255, s, v


def rgb2hsi(img, img_path):
    img = normalize2(img)
    # img = img.astype(int)
    m, n, _ = img.shape
    output_img = np.zeros((m, n, 3), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            output_img[i, j] = pix_hsi(
                img[i, j, 0], img[i, j, 1], img[i, j, 2])
    output_path = save_output(
        output_img[:, :, [2, 1, 0]], img_path, 'HSI', split=True)
    return output_path


def rgb2xyz(img, img_path, save=True):
    img = normalize2(img)
    r, g, b = cv2.split(img)
    output_img = np.zeros(img.shape, dtype=np.uint8)
    output_img[:, :, 0] = 255*(0.412453*r + 0.357580*g + 0.180423*b)
    output_img[:, :, 1] = 255*(0.212671*r + 0.715160*g + 0.072169*b)
    output_img[:, :, 2] = 255*(0.019334*r + 0.119193*g + 0.950227*b)
    if save:
        output_path = save_output(
            output_img[:, :, [2, 1, 0]], img_path, 'XYZ', split=True)
        return output_path
    return output_img


def rgb2lab(img, img_path):
    xyz = rgb2xyz(img, img_path, save=False)
    xyz = normalize2(xyz)
    x, y, z = cv2.split(xyz)
    xn = 0.412453 + 0.357580 + 0.180423
    yn = 0.212671 + 0.715160 + 0.072169
    zn = 0.019334 + 0.119193 + 0.950227
    x /= xn
    y /= yn
    z /= zn
    l = np.where(y > 0.008856, 116*pow(y, 1/3)-16, 903.3*y)
    a = 500 * np.where(x > 0.008856, pow(x, 1/3)-pow(y, 1/3), 7.787*(x-y))
    b = 200 * np.where(y > 0.008856, pow(y, 1/3)-pow(z, 1/3), 7.787*(y-z))
    output_img = normalize2(np.stack((l, a, b), axis=2))*255
    output_img = output_img.astype(np.uint8)
    # return output_img
    output_path = save_output(
        output_img[:, :, [2, 1, 0]], img_path, 'Lab', split=True)
    return output_path


def rgb2yuv(img, img_path):
    img = normalize2(img)
    r, g, b = cv2.split(np.float32(img))
    output_img = np.zeros(img.shape, dtype=np.uint8)
    output_img[:, :, 0] = 255*(0.299*r + 0.587*g + 0.114*b)
    output_img[:, :, 1] = 255*(-0.169*r - 0.331*g + 0.5*b + 128)
    output_img[:, :, 2] = 255*(0.5*r - 0.419*g - 0.081*b + 128)
    output_path = save_output(
        output_img[:, :, [2, 1, 0]], img_path, 'YUV', split=True)
    return output_path


def pseudo(img, level, color_start, color_end, img_path):
    level = level - 1
    r, g, b = cv2.split(np.float32(img))
    rs = color_start.red()
    gs = color_start.green()
    bs = color_start.blue()
    re = color_end.red()
    ge = color_end.green()
    be = color_end.blue()
    gray = 0.299*r + 0.587*g + 0.114*b
    stepr = (re-rs)/level
    stepg = (ge-gs)/level
    stepb = (be-bs)/level
    output_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output_img[i, j, 0] = np.around(
                rs + np.round(gray[i, j]*level/255)*stepr)
            output_img[i, j, 1] = np.around(
                gs + np.round(gray[i, j]*level/255)*stepg)
            output_img[i, j, 2] = np.around(
                bs + np.round(gray[i, j]*level/255)*stepb)
    output_path = save_output(
        output_img[:, :, [2, 1, 0]], img_path, 'RGB', split=True)
    return output_path


def segmentation(img, k, img_path):
    # reshape to m*n 3-d points.
    img_point = img.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, label, center = cv2.kmeans(
        np.float32(img_point), k, None, criteria, 10, flags)
    center = np.uint8(center)
    result = center[label.flatten()]
    output_img = result.reshape((img.shape))
    output_path = save_output(
        output_img[:, :, [2, 1, 0]], img_path, 'RGB', split=True)
    return output_path


def normalize2(img):
    normal = img - img.min()
    return normal/normal.max()


if __name__ == '__main__':
    img_path = './HW05-Part 3-02.bmp'
    img = cv2.imread(img_path)
    plt.subplot(1, 3, 1)
    # plt how in RGB format
    plt.imshow(img[:, :, [2, 1, 0]])

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))

    from PyQt5.QtGui import QColor
    color1 = QColor(255, 0, 0)
    color2 = QColor(0, 0, 255)

    output = pseudo(img[:, :, [2, 1, 0]], 2, color1, color2, img_path)
    # output = rgb2lab(img[:, :, [2, 1, 0]], img_path)
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.show()


# c=np.array(range(24)).reshape((2,4,3))
