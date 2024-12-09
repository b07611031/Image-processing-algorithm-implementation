import numpy as np
import random
import math


def gsolve(Z, B, l, w):
    n = 256
    A = np.empty((Z.shape[0]*Z.shape[1]+n+1, n+Z.shape[0]))
    b = np.empty(A.shape[0])
    k = 1
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i][j]]
            A[k-1][Z[i][j]] = wij
            A[k-1][n+i] = -wij
            b[k-1] = wij * B[j]
            k = k+1
    A[k-1][128] = 1
    k = k+1
    for i in range(n-1):
        A[k-1][i] = l*w[i+1]
        A[k-1][i+1] = -2*l*w[i+1]
        A[k-1][i+2] = l*w[i+1]
        k = k+1
    x = np.linalg.pinv(A).dot(b)
    g = x[:256]
    le = x[256:]
    return g, le


def debevec_hdr(imgCell, exposureTimes, n=500, name='exp'):
    
    exposureTimes = np.array(exposureTimes, dtype=np.float32)
    height, width, _ = imgCell[0].shape
    x_list = []
    y_list = []
    numPics = len(imgCell)
    for i in range(n):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        x_list.append(x)
        y_list.append(y)
    Zr = np.array([[0]*(numPics)]*(n))
    Zg = np.array([[0]*(numPics)]*(n))
    Zb = np.array([[0]*(numPics)]*(n))
    for i in range(n):
        x = x_list[i]
        y = y_list[i]
        for j in range(numPics):
            img = imgCell[j]
            Zr[i][j] = img[y, x, 0]
            Zg[i][j] = img[y, x, 1]
            Zb[i][j] = img[y, x, 2]
    B = np.log(exposureTimes)
    l = 1
    w = np.zeros(256)
    for i in range(256):
        if(i < 128):
            w[i] = i+1
        else:
            w[i] = 256-i
    [gr, lEr] = gsolve(Zr, B, l, w)
    [gg, lEg] = gsolve(Zg, B, l, w)
    [gb, lEb] = gsolve(Zb, B, l, w)
    gcell = np.array([gr, gg, gb])
    hdrImg = np.zeros((height, width, 3))

    for c in range(3):
        for i in range(height):
            for j in range(width):
                wij = 0
                lEg = 0
                for k in range(1, numPics):
                    lE = gcell[c][imgCell[k][i][j][c]] - B[k]
                    lEg = w[imgCell[k][i][j][c]]*lE + lEg
                    wij = wij + w[imgCell[k][i][j][c]]
                lEg = lEg/wij
                hdrImg[i][j][c] = math.exp(lEg)

    import matplotlib.pyplot as plt
    import os
    plt.plot(gr, 'r', label='R')  # black dashed line, with "+" markers
    plt.plot(gg, 'g', label='G')     # green dimonds (no line)
    plt.plot(gb, 'b', label='B')    # red dotted line (no marker)
    plt.legend(loc='best', shadow=True, facecolor='#ccc', edgecolor='#000')
    plt.title("g function with {} points".format(n), {
              'fontsize': 15})     # red dotted line (no marker)
    os.makedirs('./output', exist_ok=True)
    plt.savefig(f'./output/{name}_g.png')
    print(f'The g function graph has been saved to ./output/{name}_g.png.')
    # plt.show()
    return hdrImg