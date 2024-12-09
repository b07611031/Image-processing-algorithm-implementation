import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pywt
from scipy.signal import convolve2d as conv2d
import time


def save_output(img, img_path, suffix=''):
    img[img > 255] = 255
    img[img < 0] = 0
    os.makedirs('./output', exist_ok=True)
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + \
        '_' + suffix + '.' + file_name[-1]
    cv2.imwrite(save_path, img)
    return save_path


# Luma Grayscale
def gray_luma(img, img_path='Output.jpg', suffix=''):
    os.makedirs('./output', exist_ok=True)
    gray = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + suffix + '_gray.' + file_name[-1]
    cv2.imwrite(save_path, gray)
    # return save_path
    return gray


def trapezoidal(img, img_path):
    m, n = img.shape
    output_img = np.zeros(img.shape)
    vertical = 3/4
    horizontal = 1/2
    h_shift = (1-horizontal)/2
    for i in range(m):
        for j in range(n):
            a = round(vertical*i)
            b = round(j + i*h_shift - i*j*horizontal/n)
            output_img[a, b] = img[i, j]
    output_path = save_output(
        output_img, img_path, 'trapezoidal')
    return output_path


def wavy(img, img_path):
    m, n = img.shape
    output_img = np.zeros(img.shape)
    wave = 50
    freq = 1/200
    for i in range(m):
        for j in range(n):
            a = round(i - wave*np.sin(2*np.pi*j*freq))
            b = round(j - wave*np.sin(2*np.pi*i*freq))
            if (a >= 0) and (a < m) and (b >= 0) and (b < n):
                output_img[i, j] = img[a, b]
    output_path = save_output(
        output_img, img_path, 'wavy')
    return output_path


def circular(img, img_path):
    m, n = img.shape
    output_img = np.zeros(img.shape)
    m_cent = m/2
    for i in range(1, m):
        for j in range(1, n):
            a = i
            new_cent = np.sqrt(pow(m_cent, 2) - pow(m_cent-i, 2))
            b = round((j-m_cent)*m_cent/new_cent + m_cent)
            if (a >= 0) and (a < m) and (b >= 0) and (b < n):
                output_img[i, j] = img[a, b]
    output_path = save_output(
        output_img, img_path, 'circular')
    return output_path


def fusion(img1, img2, img3, img_path):
    m1, n1 = img1.shape
    m2, n2 = img2.shape
    if m1*n1 <= m2*n2:
        img2 = my_resize(img2, m1, n1)
    else:
        img1 = my_resize(img2, m2, n2)
    coeffs1 = pywt.dwt2(img1, 'haar')
    ll1, (lh1, hl1, hh1) = coeffs1
    coeffs2 = pywt.dwt2(img2, 'haar')
    ll2, (lh2, hl2, hh2) = coeffs2
    if img3 is None:
        ll = (ll1+ll2)/2
        lh = np.maximum(lh1, lh2)
        hl = np.maximum(hl1, hl2)
        hh = np.maximum(hh1, hh2)
        coeffs = ll, (lh, hl, hh)
        output_img = pywt.idwt2(coeffs, 'haar')
    else:
        img3 = my_resize(img3, img1.shape[0], img1.shape[1])
        coeffs3 = pywt.dwt2(img3, 'haar')
        ll3, (lh3, hl3, hh3) = coeffs3
        ll = (ll1+ll2+ll3)/2
        lh = np.maximum(lh1, lh2, lh3)
        hl = np.maximum(hl1, hl2, hl3)
        hh = np.maximum(hh1, hh2, hh3)
        coeffs = ll, (lh, hl, hh)
        output_img = pywt.idwt2(coeffs, 'haar')
    output_path = save_output(
        output_img, img_path, 'fusion')
    return output_path


def my_resize(img, height, width):
    m, n = img.shape
    x_ratio = float(n - 1) / (width - 1)
    y_ratio = float(m - 1) / (height - 1)
    img = img.ravel()

    x = np.tile(np.arange(width), height)
    y = np.repeat(np.arange(height), width)
    x_l = np.floor(x_ratio * x).astype('int32')
    x_h = np.ceil(x_ratio * x).astype('int32')
    y_l = np.floor(y_ratio * y).astype('int32')
    y_h = np.ceil(y_ratio * y).astype('int32')
    x_weight = (x_ratio * x) - x_l
    y_weight = (y_ratio * y) - y_l

    a = img[y_l * n + x_l]
    b = img[y_l * n + x_h]
    c = img[y_h * n + x_l]
    d = img[y_h * n + x_h]
    resized = a * (1 - x_weight) * (1 - y_weight) + \
        b * x_weight * (1 - y_weight) + \
        c * y_weight * (1 - x_weight) + \
        d * x_weight * y_weight
    return resized.reshape(height, width)


def padding(img, kernel):
    padding_size = int((kernel.shape[0]-1)/2)
    img_padding = np.zeros(
        (img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
    img_padding[padding_size:-padding_size,
                padding_size:-padding_size] = img
    return img_padding


def conv_2d(img_arr, kernel):
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    img_padding = padding(img_arr, kernel)
    img_conv = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            window = kernel * img_padding[i:i +
                                          kernel.shape[0], j:j+kernel.shape[1]]
            img_conv[i, j] = window.sum()
    return img_conv.astype(np.int8)


def hough(img, img_path):
    output_img = cv2.imread(img_path)
    m, n = img.shape
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_img = np.sqrt((np.power(conv2d(img, gx), 2)) +
                        (np.power(conv2d(img, gy), 2)))

    d = round(2 * np.sqrt(pow(m, 2)+pow(n, 2)))
    h_space = np.zeros((d, 180))
    for i in range(m):
        for j in range(n):
            if sobel_img[i, j] >= 127:
                for theta in range(-90, 90):
                    rho = round(i*np.cos(np.radians(theta)) + j *
                                np.sin(np.radians(theta)))+int(d/2)
                    h_space[rho, theta+89] = h_space[rho, theta+89] + 1

    line = []
    threshold = h_space.max()/4
    coef = m * n
    for i in range(d):
        for j in range(180):
            if h_space[i, j] >= threshold:
                line.append((i, j))
                rho = i - np.sqrt(pow(m, 2)+pow(n, 2))
                theta = j - 89
                a = np.sin(np.radians(theta))
                b = np.cos(np.radians(theta))
                x1 = int(a*rho - b*coef)
                y1 = int(b*rho + a*coef)
                x2 = int(a*rho + b*coef)
                y2 = int(b*rho - a*coef)
                cv2.line(output_img, (x1, y1), (x2, y2), (255, 140, 0), 2)
    output_path = save_output(
        output_img, img_path, 'hough')
    return output_path
    # return output_img


if __name__ == '__main__':
    img_path = './Part 3 Image/rects.bmp'
    # img_path = './Part 3 Image/square.jpg'
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    # img = gray_luma(img, img_path)
    # ouput_path = circular(img, img_path)

    # img1 = cv2.imread(
    #     './mart 2 Images/Image Set 2/multifocus1.JmG', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(
    #     './mart 2 Images/Image Set 2/multifocus2.JmG', cv2.IMREAD_GRAYSCALE)
    # img3 = cv2.imread(
    #     './mart 2 Images/Image Set 2/multifocus3.JmG', cv2.IMREAD_GRAYSCALE)
    # output_path = fusion(img1, img2, img3)
    # plt.imshow(cv2.imread(output_path))
    # plt.imshow(my_resize(img1, img.shape[0], img.shape[0]))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h_path = hough(img, img_path)
    plt.imshow(cv2.imread(h_path))
    plt.show()
