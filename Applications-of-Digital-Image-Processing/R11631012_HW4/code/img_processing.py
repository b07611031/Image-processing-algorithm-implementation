import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time


def plot_hist(img, img_path='Output_hist.jpg', suffix=''):
    os.makedirs('./output', exist_ok=True)
    save_path = './output/' + \
        img_path.split('/')[-1].split('.')[0] + suffix + '_hist.png'
    img = img.real
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


def save_output(img, img_path, suffix=''):
    img[img > 255] = 255
    img[img < 0] = 0
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + \
        '_' + suffix + '.' + file_name[-1]
    cv2.imwrite(save_path, img)
    return save_path


# Average Grayscale
def gray_average(imgA):
    return imgA[:, :, 0]/3 + imgA[:, :, 1]/3 + imgA[:, :, 2]/3


# Luma Grayscale
def gray_luma(img, img_path='Output.jpg', suffix=''):
    os.makedirs('./output', exist_ok=True)
    gshiftray = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + suffix + '_gray.' + file_name[-1]
    cv2.imwrite(save_path, gshiftray)
    return gshiftray, save_path


def normalize2(img):
    delta = img.max()-img.min()
    normal = img - img.min()
    normal = normal * (255/delta)
    return normal
    # img = np.clip(img, 0, 1)
    # return img*255


def fft2(img):
    img = img/img.max()
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def fft(img, img_path):
    time_start = time.time()
    fshift = fft2(img)
    fft_time = time.time()-time_start
    spectrum = np.abs(fshift)
    fmin = np.log(1+np.abs(spectrum.min()))
    fmax = np.log(1+np.abs(spectrum.max()))
    ynew = 255*(np.log(1+abs(spectrum))-(fmin))/(fmax-fmin)
    spectrum_path = save_output(ynew, img_path, 'spectrum')
    phase = np.arctan(fshift.imag/fshift.real)
    phase_path = save_output(normalize2(phase), img_path, 'phase')
    return fshift, spectrum_path, phase_path, fft_time


def ifft(fshift, img_path='./', save=False):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = normalize2(img_back)
    if save:
        save_path = save_output(img_back.real, img_path, 'back')
        return img_back.real, save_path
    return img_back.real


def fft_d(img_arr, fshift, img_path):
    img_back = ifft(fshift)
    img_d = img_arr - img_back.real
    img_d = normalize2(img_d)
    save_path = save_output(img_d, img_path, 'd')
    return img_d.real, save_path


def ideal_filter(fshift, cutoff, f_pass, img_path):
    p, q = fshift.shape
    filter_h = np.zeros((p, q), dtype='complex_')
    # Lowpass filter
    if f_pass == 'Lowpass':
        for i in range(p):
            for j in range(q):
                distance = pow((pow(i-p/2, 2)+pow(j-q/2, 2)), 0.5)
                if distance <= cutoff:
                    filter_h[i][j] = 1
    # Highpass filter
    elif f_pass == 'Highpass':
        for i in range(p):
            for j in range(q):
                distance = pow((pow(i-p/2, 2)+pow(j-q/2, 2)), 0.5)
                if distance > cutoff:
                    filter_h[i][j] = 1
    gshift = fshift * filter_h
    img_back = ifft(gshift)
    img_back = normalize2(img_back)
    save_path = save_output(img_back, img_path, 'ideal')
    return img_back, save_path


def gaussian_filter(fshift, cutoff, f_pass, img_path):
    p, q = fshift.shape
    filter_h = np.zeros((p, q), dtype='complex_')
    # Lowpass filter
    for i in range(p):
        for j in range(q):
            distance = pow((pow(i-p/2, 2)+pow(j-q/2, 2)), 0.5)
            filter_h[i, j] = np.exp((-pow(distance, 2))/(2*pow(cutoff, 2)))
    # Highpass filter
    if f_pass == 'Highpass':
        filter_h = 1-filter_h
    gshift = fshift * filter_h
    img_back = ifft(gshift)
    img_back = normalize2(img_back)
    save_path = save_output(img_back, img_path, 'gaussian')
    return img_back, save_path


def butterworth_filter(fshift, cutoff, n, f_pass, img_path):
    p, q = fshift.shape
    filter_h = np.zeros((p, q), dtype='complex_')
    # Lowpass filter
    for i in range(p):
        for j in range(q):
            distance = pow((pow(i-p/2, 2)+pow(j-q/2, 2)), 0.5)
            filter_h[i, j] = 1/(1+pow(distance/cutoff, 2*n))
    # Highpass filter
    if f_pass == 'Highpass':
        filter_h = 1-filter_h
    gshift = fshift * filter_h
    img_back = ifft(gshift)
    img_back = normalize2(img_back)
    save_path = save_output(img_back, img_path, 'butter')
    return img_back, save_path


def homo_filter(fshift, gh, gl, d0, img_path):
    p, q = fshift.shape
    filter_h = np.zeros((p, q), dtype='complex_')
    c = 5
    for i in range(p):
        for j in range(q):
            distance2 = (pow(i-p/2, 2)+pow(j-q/2, 2))
            filter_h[i, j] = (gh-gl)*(1-np.exp(-c*distance2/pow(d0, 2)))+gl
    gshift = fshift * filter_h
    img_back = ifft(gshift)
    img_back = normalize2(img_back)
    save_path = save_output(img_back, img_path, 'homo')
    return img_back, save_path


def noise_filter(img_arr, mean, sigma, img_path, save=True):
    noise = np.random.normal(mean, sigma, img_arr.shape)
    noise = (noise-mean)/(2.5*sigma)
    gaussian_out = np.clip(noise, 0, 1)
    gaussian_out *= 255
    if save:
        img_back = img_arr+gaussian_out
        img_back = normalize2(img_back)
        save_path = save_output(img_back, img_path, 'noise')
        return img_back, save_path
    # img_arr is gshift if save=False
    gau_shift = fft2(gaussian_out)
    return (img_arr+gau_shift)


def motion_filter(fshift, a, b, t, img_path, save=True):
    p, q = fshift.shape
    filter_h = np.zeros((p, q), dtype='complex128')
    for u in range(p):
        for v in range(q):
            d = np.pi * (u*a+v*b)
            if d == 0:
                d = 1
            # d = np.pi * ((u+1)*a/10+(v+1)*b/30)
            filter_h[u, v] = (t/d) * np.sin(d)*np.exp(-1j*d)
    hshift = np.fft.fftshift(filter_h)
    gshift = fshift * hshift
    if save:
        img_back = ifft(gshift)
        img_back = normalize2(np.abs(img_back))
        save_path = save_output(img_back, img_path, 'motion')
        return img_back, save_path
    return gshift, hshift


def unblur_filter(gshift, way, hshift, K, img_path):
    if way == 'Inverse filter':
        f_hat = gshift/hshift
    elif way == 'Wiener filter':
        h2 = (hshift.imag)*(hshift.real)
        f_hat = ((1/hshift)*h2/(h2+K))*gshift
    img_back = ifft(f_hat)
    img_back = normalize2(img_back)
    save_path = save_output(img_back, img_path, 'unblur')
    return img_back, save_path


def unblur_d_filter(gshift, hshift, K, img_path):
    f_inverse = gshift/hshift
    h2 = (hshift.imag)*(hshift.real)
    f_wiener = (1/hshift*h2/(h2+K))*gshift
    f_hat = f_wiener - f_inverse
    img_back = ifft(f_hat)
    img_back = normalize2(img_back)
    save_path = save_output(img_back, img_path, 'unblurD')
    return img_back, save_path


def motion_unblur_filter(gshift, a, b, t, way, K, img_path):
    _, hshift = motion_filter(gshift, a, b, t, img_path, save=False)
    if way == 'Inverse filter':
        f_hat = gshift/hshift
    elif way == 'Wiener filter':
        h2 = (hshift.imag)*(hshift.real)
        f_hat = (1/hshift*h2/(h2+K))*gshift
    img_back = ifft(f_hat)
    img_back = normalize2(img_back)
    save_path = save_output(img_back, img_path, 'restore')
    return img_back, save_path