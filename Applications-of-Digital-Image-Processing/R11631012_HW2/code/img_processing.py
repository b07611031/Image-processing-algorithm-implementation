import numpy as np
import matplotlib.pyplot as plt


def plot_hist(img, img_path='Output.jpg', suffix=''):
    img = np.around(img.ravel()).astype(int)
    labels, counts = np.unique(img, return_counts=True)
    y = np.zeros(256)
    for index, num in zip(labels, counts):
        y[index] = num
    x = range(0, 256)
    plt.bar(x, y, align='center')
    save_path = img_path.split('/')[-1].split('.')[0] + suffix + '_hist.png'
    # print(save_path)
    plt.savefig(save_path)
    # plt.show()
    plt.clf()
    return save_path

def _hist(img):
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i,j]]+=1
    return values

def _cdf(hist):
    cdf = [0] * len(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i]= cdf[i-1]+hist[i]

    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf

def imequalize(img):
    my_cdf = _cdf(_hist(img.astype(np.int)))
    processed = np.interp(img, range(0,256), my_cdf)
    return processed

# from PIL import Image
# img = read_b64("LISA.64")
# plot_hist(img)

# img_path='Output.jpg'
# save_path = img_path.split('/')[-1].split('.')[0] + '_hist.png'
# print(save_path)