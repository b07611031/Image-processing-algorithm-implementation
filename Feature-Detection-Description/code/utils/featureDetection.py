import cv2
import numpy as np
import time


def HarrisCorner(img, nm='argmax', n=50):
    htime = time.time()
    img_gray = grayscale(img, 'luma')
    rsp = harris(img_gray)
    print(f'Harris takes {time.time()-htime}s')
    ftx, fty = nonmax(rsp, nm=nm, n=n)
    print(f'All takes {time.time()-htime}s')
    return rsp, ftx, fty


def multiscaleHarris(img, n=100, des=False, nm='argmax'):
    htime = time.time()
    scale = 3
    img0 = grayscale(img, 'luma')
    imgs = [img0]
    rsp = harris(img0)
    scaleMap = np.zeros(rsp.shape)
    for l in range(1, scale):
        # print(l)
        imgs.append(downsample2(imgs[l-1]))
        rspi = harris(imgs[l])
        px, py = np.where(rspi > 0)
        for pt in zip(px, py):
            s = np.power(2, l)
            if rspi[pt[0], pt[1]] > rsp[pt[0]*s, pt[1]*s]:
                rsp[pt[0]*s, pt[1]*s] = rspi[pt[0], pt[1]]
                scaleMap[pt[0]*s, pt[1]*s] = l
    if nm =='argmax':
        ftx, fty = nonmax(rsp, nm='argmax', n=n)
    elif nm == 'anms':
        ftx, fty = nonmax(rsp, nm='anms', n=n)
    else:
        return print('"argmax", or "anms')
    if des:
        return imgs, scaleMap, ftx, fty
    print(f'All takes {time.time()-htime}s')
    return rsp, ftx, fty


def harris(img):
    # img is grayscale
    img = gaussian_filter(img, kernel_size=3, sigma=1.0)
    ix, iy = sobel_filter(img)
    sigma = 1.5
    sx2 = gaussian_filter(np.square(ix), kernel_size=3, sigma=sigma)
    sy2 = gaussian_filter(np.square(iy), kernel_size=3, sigma=sigma)
    sxy = gaussian_filter(ix*iy, kernel_size=3, sigma=sigma)
    rsp = (sx2*sy2-np.square(sxy)) / (sx2+sy2+1e-8)
    rsp = localmax(rsp, 3)
    return rsp
    # rsp = (sx2*sy2-np.square(sxy)) - 0.4*(sx2+sy2)
    # rspheat = normalize2(rsp)
    # return rspheat


def localmax(rsp, k=3):
    img_padding = padding(rsp, k)
    rsp_max = np.zeros(rsp.shape)
    for i in range(rsp.shape[0]):
        for j in range(rsp.shape[1]):
            window = img_padding[i:i+k, j:j+k]
            if rsp[i, j] == window.max() and rsp[i, j] >= 10:
                rsp_max[i, j] = rsp[i, j]
    return rsp_max


def downsample2(img):
    return img[::2, ::2]


def anms(rsp, n=500):
    argrsp = np.argsort(rsp, axis=None)[::-1]
    ftxr, ftyr = np.divmod(argrsp, rsp.shape[1])
    ftx, fty = [ftxr[0]], [ftyr[0]]
    r = np.sqrt(np.square(rsp.shape[0])+np.square(rsp.shape[1])).astype(int)
    while (len(ftx) < n and r >= 1):
        # print(r)
        outx = ftxr.copy()
        outy = ftyr.copy()
        for pt in zip(ftx, fty):
            for pt_in in np.where(np.logical_and(outx <= pt[0]+r, outx >= pt[0]-r))[0]:
                if np.logical_and(outy[pt_in] <= pt[1]+r, outy[pt_in] >= pt[1]-r):
                    outx[pt_in] = 0
                    outy[pt_in] = 0
        r = r - 1
        if len(np.flatnonzero(np.logical_and(outx, outy))) == 0:
            continue
        outmax = np.flatnonzero(np.logical_and(outx, outy))[0]
        if rsp[ftxr[outmax], ftyr[outmax]] > 0:
            r += 1
            ftx.append(ftxr[outmax])
            fty.append(ftyr[outmax])
    print(f'Final r is {r}.')
    return ftx, fty


def nonmax(rsp, nm='', n=50):
    if nm == 'thr':
        ftx, fty = np.where(rsp >= 0.8, rsp, 0)
    elif nm == 'argmax':
        argrsp = np.argsort(rsp, axis=None)[::-1][:n]
        ftx, fty = np.divmod(argrsp, rsp.shape[1])
    elif nm == 'anms':
        ftx, fty = anms(rsp, n=n)

    return ftx, fty


def normalize2(img):
    normal = img - img.min()
    return normal/(normal.max()+1e-8)


def sobel_filter(img_arr):
    hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_gx = conv_2d(img_arr, hx)
    img_gy = conv_2d(img_arr, hy)
    # img_sobel = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    # img_sobel = np.sqrt((np.power(img_gx, 2))+(np.power(img_gy, 2)))
    # return img_sobel
    return img_gx, img_gy


def gaussian_filter(img_arr, kernel_size=3, sigma=1, k=1):
    c = int((kernel_size-1)/2)
    distance = np.fromfunction(lambda i, j:
                               pow((i-c), 2)+pow((j-c), 2), (kernel_size, kernel_size), dtype=float)
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = k*np.exp(-distance[i][j]/(2*pow(sigma, 2)))
    kernel /= kernel.sum()
    img_gaussian = conv_2d(img_arr, kernel)
    return img_gaussian


def padding(img, kernelsize):
    padding_size = int((kernelsize-1)/2)
    img_padding = np.zeros(
        (img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
    img_padding[padding_size:-padding_size,
                padding_size:-padding_size] = img
    return img_padding


def conv_2d(img_arr, kernel):
    # time_start = time.time()
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    img_padding = padding(img_arr, kernel.shape[0])
    img_conv = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            window = kernel * img_padding[i:i+kernel.shape[0],
                                          j:j+kernel.shape[1]]
            img_conv[i, j] = window.sum()
    img_conv[img_conv > 255] = 255
    img_conv[img_conv < 0] = 0
    # print(f'Convolution takes {time.time()-time_start} seconds.')
    return img_conv


def grayscale(img, strtg='grey'):
    # The input image is a 3-dim RGB image.
    if strtg == 'grey':
        return 54/256*img[:, :, 0] + 183/256*img[:, :, 1] + 19/256*img[:, :, 2]
    elif strtg == 'luma':
        return 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    elif strtg == 'green':
        return img[:, :, 1]


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    img_path = '../test/cat.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # while max(img.shape) >1000:
    #     img = cv2.resize(img, (int(img.shape[0]/2), int(img.shape[1]/2)))
    print(img.shape)
    # argmax, anms
    # rsp, ftx, fty = HarrisCorner(img, nm='argmax', n=100)
    rsp, ftx, fty = multiscaleHarris(img, n=250, nm='argmax')
    rsp = cv2.applyColorMap(np.uint8(normalize2(rsp)*254+1), cv2.COLORMAP_JET)
    print(len(ftx))

    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 3, 1)
    # plt.axis('off')
    # plt.imshow(img)
    # plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(rsp, cmap=plt.cm.jet)
    # plt.subplot(1, 3, 3)
    # plt.axis('off')
    # plt.imshow(img)
    # for pt in zip(ftx, fty):
    #     plt.plot(pt[1], pt[0], 'r*')
    plt.savefig('multiHarris.png')
    # plt.savefig('multiHarris_anms_250.png')
    # plt.savefig('multiHarris_max_100.png')
    plt.show()
    
    # plt.figure(figsize=(12, 6))
    # plt.axis('off')
    # plt.imshow(img)
    # for pt in zip(ftx, fty):
    #     plt.plot(pt[1], pt[0], 'r*')
    # plt.savefig('cat_multiHarris_anms_250.png')
    # plt.savefig('cat_multiHarris_max_100.png')
    # plt.show()
