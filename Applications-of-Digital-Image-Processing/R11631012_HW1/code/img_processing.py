import numpy as np
import matplotlib.pyplot as plt


def read_b64(filename):
    b64_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15,
                'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
                'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31}
    
    with open(filename, "r") as f:
        b64 = f.readlines()
    b64 = [x.strip('\n').strip('\x1a') for x in b64 if x != '\x1a']
    # print(b64[-1])
    lst = [[b64_dict[x] for x in row] for row in b64]
    img = np.array(lst, dtype=float)*255/31
    # img = cv.fromarray(img)
    # cv.imshow('arr', img, cv.IMREAD_GRAYSCALE)
    return img

def plot_hist(img, img_path='Output.64'):
    img = np.around(img.ravel()*31/255).astype(int)
    labels, counts = np.unique(img, return_counts=True)
    y = np.zeros(32)
    for index, num in zip(labels, counts):
        y[index] = num
    x = range(0, 32)
    plt.bar(x, y, align='center')
    save_path = img_path.split('/')[-1].replace(".64", "_hist.png")
    plt.savefig(save_path)
    # plt.show()
    plt.clf()
    return save_path
    
# from PIL import Image
# img = read_b64("LISA.64")
# plot_hist(img)