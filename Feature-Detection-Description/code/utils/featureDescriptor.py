import featureDetection as fdet
import cv2
import numpy as np


class featureVector:
    def __init__(self, scale, orientation, patch):
        self.scale = scale
        self.orientation = orientation
        self.patch = patch


def descriptor(imgs, scaleMap, ftx, fty):
    # imgs is a image set contain images of different scales.
    blurs = [fdet.gaussian_filter(img, kernel_size=3, sigma=4.5) for img in imgs]
    oriMaps = []
    for img in blurs:
        ix, iy = fdet.sobel_filter(img)
        # using image corrdinate system. x-down, y-right. oriMap in radian.
        oriMaps.append(np.arctan(ix/(iy+1e-8)))
    
    featureVectors = []
    for i in range(len(ftx)):
        l = scaleMap[ftx[i], fty[i]].astype(int)
        fx = ftx[i]/np.power(2, l)
        fy = fty[i]/np.power(2, l)
        angle = oriMaps[l][int(fx), int(fy)]
        patch = findPatch(blurs[l], fx, fy, angle)
        featureVectors.append(patch2Haar(patch).ravel())
    return np.array(featureVectors)
    


def patch2Haar(patch):
    h8 = haarMatrix(8)
    return np.matmul(h8.T, np.matmul(patch, h8))


def findPatch(img, fx, fy, angle):
    p_size = 40
    patch = np.zeros((p_size, p_size))
    for i in range(p_size):
        for j in range(p_size):
            x0 = i - p_size/2
            y0 = j - p_size/2
            angle0 = np.arctan(y0/(x0+1e-8))
            angle0 += np.pi if x0 < 0 else 0
            r = np.sqrt(x0*x0+y0*y0)
            x1 = round(r*np.cos(angle0+angle) + fx)
            y1 = round(r*np.sin(angle0+angle) + fy)
            if 0 <= x1 < img.shape[0] and 0 <= y1 < img.shape[1]:
                patch[i, j] = img[x1, y1]
#     patch = patch[::5, ::5]
    patch = cv2.resize(patch, (8, 8))
    patch = (patch-np.mean(patch))/np.std(patch)
    return patch


def haarMatrix(n, normalized=False):
    # reference from wiki
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])
    h_n = np.kron(h, [1, 1])
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    h = np.vstack((h_n, h_i))
    return h


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    # os.chdir('./code')
    img_paths = ['../test/cat.jpg']
    imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            for img_path in img_paths]
    # rsp, ftx, fty = fdet.multiscaleHarris(imgs[0], n=1, des=False, nm='argmax')
    # fx = ftx[0]
    # fy = fty[0]
    
    fx=1200
    fy=1000
    img0 = fdet.grayscale(imgs[0], 'luma')
    # patch1 = findPatch(img0, fx, fy, np.radians(45))
    patch2 = findPatch(img0, fx, fy, np.radians(125))

    # print(patch1.shape)
    # print(patch2Haar(patch1).shape)
    # print(patch2Haar(patch1))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.plot(fy, fx, 'wo', markersize=3)
    plt.plot([fy, fy-130], [fx, fx-180], 'w-')
    plt.imshow(imgs[0])
    plt.subplot(1, 3, 2)
    plt.axis('off')
    ps2 = 20
    plt.imshow(img0[fx-ps2:fx+ps2, fy-ps2:fy+ps2], cmap='gray')
    plt.plot(ps2, ps2, 'wo')
    plt.plot([ps2, ps2-13], [ps2, ps2-18], 'w-')
    plt.subplot(1, 3, 3)
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(patch2, cmap='gray')
    c=3.5
    plt.plot([c, 7.5], [c-0.05, c-0.05], 'k-')
    plt.plot([c, 7.5], [c+0.05, c+0.05], 'k-')
    plt.plot(c, c, 'wo', lw=1)
    plt.plot([c, 7.5], [c, c], 'w-')
    plt.savefig('cat_patch.png')
    plt.show()

"""
pywt.dwt2(np.array(range(64)).reshape((8,8,)), 'haar')[0].ravel()
"""
