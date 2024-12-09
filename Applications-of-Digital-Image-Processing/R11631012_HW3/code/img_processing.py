import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time


def plot_hist(img, img_path='Output_hist.jpg', suffix=''):
    os.makedirs('./output', exist_ok=True)
    save_path = './output/' + \
        img_path.split('/')[-1].split('.')[0] + suffix + '_hist.png'
    img = np.around(img.ravel()).astype(int)
    labels, counts = np.unique(img, return_counts=True)
    y = np.zeros(256)
    for index, num in zip(labels, counts):
        y[index] = num
    x = range(0, 256)
    plt.bar(x, y, align='center')
    plt.savefig(save_path)
    plt.clf()
    return save_path


# Average Grayscale
def gray_average(imgA):
    return imgA[:, :, 0]/3 + imgA[:, :, 1]/3 + imgA[:, :, 2]/3


# Luma Grayscale
def gray_luma(img, img_path='Output.jpg', suffix=''):
    os.makedirs('./output', exist_ok=True)
    img_gray = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + suffix + '_gray.' + file_name[-1]
    cv2.imwrite(save_path, img_gray)
    return img_gray, save_path


def convolution(img, kernel, img_path, suffix=''):
    img_conv = conv_2d(img, kernel)
    save_path = save_output(img_conv, img_path, suffix)
    return img_conv, save_path


def padding(img, kernel):
    padding_size = int((kernel.shape[0]-1)/2)
    img_padding = np.zeros(
        (img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
    img_padding[padding_size:-padding_size,
                padding_size:-padding_size] = img
    return img_padding


def conv_2d(img_arr, kernel):
    time_start = time.time()
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    img_padding = padding(img_arr, kernel)
    img_conv = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            window = kernel * img_padding[i:i+kernel.shape[0],
                                          j:j+kernel.shape[1]]
            img_conv[i, j] = window.sum()
    img_conv[img_conv > 255] = 255
    img_conv[img_conv < 0] = 0
    print(f'Convolution takes {time.time()-time_start} seconds.')
    return img_conv


def save_output(img, img_path, suffix=''):
    img[img > 255] = 255
    img[img < 0] = 0
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + \
        '_' + suffix + '_gray.' + file_name[-1]
    cv2.imwrite(save_path, img)
    return save_path


def box_filter(img_arr, kernel_size, img_path):
    kernel = np.ones((kernel_size, kernel_size))
    kernel = 1/pow(kernel_size, 2)*kernel
    img_box = conv_2d(img_arr, kernel)
    save_path = save_output(img_box, img_path, 'box')
    return img_box, save_path


def sobel_filter(img_arr, img_path):
    hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_gx = conv_2d(img_arr, hx)
    img_gy = conv_2d(img_arr, hy)
    img_sobel = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    img_sobel = np.sqrt((np.power(img_gx, 2))+(np.power(img_gy, 2)))
    save_path = save_output(img_sobel, img_path, 'sobel')
    return img_sobel, save_path


def gaussian_filter(img_arr, kernel_size, sigma, k, img_path):
    c = int((kernel_size-1)/2)
    distance = np.fromfunction(lambda i, j:
                               pow((i-c), 2)+pow((j-c), 2), (kernel_size, kernel_size), dtype=float)
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = k*np.exp(-distance[i][j]/(2*pow(sigma, 2)))
    kernel /= kernel.sum()
    img_gaussian = conv_2d(img_arr, kernel)
    save_path = save_output(img_gaussian, img_path, 'gaussian')
    return img_gaussian, save_path


def log_filter(img_arr, kernel_size, sigma, threshold, img_path):
    c = int((kernel_size-1)/2)
    distance = np.fromfunction(lambda i, j:
                               pow((i-c), 2)+pow((j-c), 2), (kernel_size, kernel_size), dtype=float)
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = ((distance[i][j]-2*pow(sigma, 2)) /
                            pow(sigma, 4))*np.exp(-distance[i][j]/(2*pow(sigma, 2)))
    kernel /= kernel.sum()
    img_log = conv_2d(img_arr, kernel)
    img_log_cross = zero_crossing(img_log, threshold)
    save_path = save_output(img_log_cross, img_path, 'LoG')
    return img_log_cross, save_path
    # save_path = save_output(img_log, img_path, 'LoG')
    # return img_log, save_path


def zero_crossing(img, threshold):
    img_crossing = img.copy()
    for i in range(1, (img.shape[0]-1)):
        for j in range(1, (img.shape[1]-1)):
            if img[i, j]*img[i+1, j] > 0 and abs(img[i, j]-img[i+1, j]) > threshold:
                img_crossing[i, j] = 255
            elif img[i, j]*img[i-1, j] > 0 and abs(img[i, j]-img[i-1, j]) > threshold:
                img_crossing[i, j] = 255
            elif img[i, j]*img[i, j+1] > 0 and abs(img[i, j]-img[i, j+1]) > threshold:
                img_crossing[i, j] = 255
            elif img[i, j]*img[i, j-1] > 0 and abs(img[i, j]-img[i, j-1]) > threshold:
                img_crossing[i, j] = 255
            else:
                img_crossing[i, j] = 0
    return img_crossing


def _hist(img):
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def _cdf(hist):
    cdf = [0] * len(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]

    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def hist_equalize(img, img_path):
    my_cdf = _cdf(_hist(img.astype(np.int)))
    processed = np.interp(img, range(0, 256), my_cdf)
    save_path = save_output(processed, img_path, 'HistEq')
    return processed, save_path


def local_enhancement(img_arr, kernel_size, k0, k1, k2, k3, cc, img_path):
    time_start = time.time()
    mean_g = img_arr.mean()
    var_g = pow(img_arr.var(), .5)
    kernel = np.zeros((kernel_size, kernel_size))
    img_padding = padding(img_arr, kernel)
    img_conv = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            window = img_padding[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            mean_kernel = window.mean()
            var_kernel = pow(window.var(), .5)
            if k0*mean_g <= mean_kernel and mean_kernel <= k1*mean_g and k2*var_g <= var_kernel and var_kernel <= k3*var_g:
                img_conv[i, j] = cc*img_arr[i, j]
            else:
                img_conv[i, j] = img_arr[i, j]
    img_conv[img_conv > 255] = 255
    img_conv[img_conv < 0] = 0
    print(f'Convolution takes {time.time()-time_start} seconds.')
    save_path = save_output(img_conv, img_path, 'Local')
    return img_conv, save_path
