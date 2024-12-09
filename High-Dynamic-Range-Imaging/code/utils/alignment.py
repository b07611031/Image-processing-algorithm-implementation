import numpy as np


def ward_alignment(imgs, img_paths, max_offset=64, thr=4, gray='grey'):
    align_paths = []
    limg = len(imgs)
    medimg = int(limg/2)
    # print(f'Images align to the {img_paths[medimg]}:')
    shiftBits = int(np.ceil(np.log2(max_offset)-1))
    gimgs = [grayscale(img, gray) for img in imgs]
    matchImg = gimgs.pop(medimg)
    align_paths.append(save_output(imgs.pop(medimg), img_paths.pop(medimg), savedir='alignment'))
    for i in range(limg-1):
        xs, ys = getExpShift(matchImg, gimgs[i], shiftBits, thr)
        # print(f'  The {img_paths[i].split("/")[-1]} shifted ({xs}, {ys}) pixels.')
        shiftedImg = imgShift(imgs[i], xs, ys)
        align_paths.append(save_output(shiftedImg, img_paths[i], savedir='alignment'))
    print('Alignment results saved to ./output/alignment/')
    return align_paths


def imgShift(img, xs, ys):
    m, n, dim = img.shape
    xs_ = abs(xs)
    ys_ = abs(ys)
    imgRet = np.zeros((m+2*xs_, n+2*ys_, dim), dtype='uint8')
    for i in range(dim):
        imgRet[(xs_+xs):(xs_+xs+m), (ys_+ys):(ys_+ys+n), i] = img[:, :, i]
    # for i in range(dim):
    #     imgRet[(xs_-xs):(xs_-xs+m), (ys_-ys):(ys_-ys+n), i] = img[:, :, i]
    return imgRet[xs_:(xs_+m), ys_:(ys_+n), :]


def getExpShift(img1, img2, shiftBits, thr):
    if (shiftBits > 0):
        smlImg1 = imgShrink2(img1)
        smlImg2 = imgShrink2(img2)
        curShift = getExpShift(smlImg1, smlImg2, shiftBits-1, thr)
        curShift *= 2
    else:
        curShift = np.array([0, 0])
    tb1, eb1 = computeBitmaps(img1, thr)
    tb2, eb2 = computeBitmaps(img2, thr)
    minErr = img1.shape[0] * img1.shape[1]
    for i in range(-1, 2):
        for j in range(-1, 2):
            xs = curShift[0] + i
            ys = curShift[1] + j
            shiftedTb2 = bitmapShift(tb2, xs, ys)
            shiftedEb2 = bitmapShift(eb2, xs, ys)
            diffB = np.bitwise_xor(tb1, shiftedTb2)
            diffB = np.bitwise_and(diffB, eb1)
            diffB = np.bitwise_and(diffB, shiftedEb2)
            err = np.sum(diffB)
            if err < minErr:
                shiftRet = np.array([xs, ys])
                minErr = err
    return shiftRet


def bitmapShift(bm, xs, ys):
    m, n = bm.shape
    xs_ = abs(xs)
    ys_ = abs(ys)
    bmRet = np.zeros((m+2*xs_, n+2*ys_), dtype=bool)
    bmRet[(xs_+xs):(xs_+xs+m), (ys_+ys):(ys_+ys+n)] = bm
    # bmRet[(xs_-xs):(xs_-xs+m), (ys_-ys):(ys_-ys+n)] = bm
    return bmRet[xs_:(xs_+m), ys_:(ys_+n)]


def computeBitmaps(img, thr):
    med = np.median(img, axis=None)
    tb = np.where(img <= med, False, True).astype(bool)
    eb = np.where(img > (med+thr), 0, img)
    eb = np.where(eb < (med-thr), False, True).astype(bool)
    return tb, eb


def imgShrink2(img):
    m, n = img.shape
    height, width = np.ceil(m/2).astype('int32'), np.ceil(n/2).astype('int32')
    x_ratio = float(n - 1) / (width - 1)
    y_ratio = float(m - 1) / (height - 1)
    img = img.ravel()

    x = np.tile(np.arange(width), height)
    y = np.repeat(np.arange(height), width)
    x_l = np.floor(x_ratio * x).astype('int32')
    y_l = np.floor(y_ratio * y).astype('int32')
    x_h = np.ceil(x_ratio * x).astype('int32')
    y_h = np.ceil(y_ratio * y).astype('int32')
    x_weight = (x_ratio * x) - x_l
    y_weight = (y_ratio * y) - y_l

    a = img[y_l*n + x_l]
    b = img[y_l*n + x_h]
    c = img[y_h*n + x_l]
    d = img[y_h*n + x_h]
    resized = a * (1 - x_weight) * (1 - y_weight) + \
        b * x_weight * (1 - y_weight) + \
        c * y_weight * (1 - x_weight) + \
        d * x_weight * y_weight
    resized -= resized.min()
    resized = resized/resized.max()*255
    return resized.astype('int32').reshape(height, width)


def grayscale(img, strtg='grey'):
    # The input image is a 3-dim RGB image.
    if strtg == 'grey':
        return 54/256*img[:, :, 0] + 183/256*img[:, :, 1] + 19/256*img[:, :, 2]
    elif strtg == 'luma':
        return 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    elif strtg == 'green':
        return img[:, :, 1]


def save_output(img, img_path, savedir='./'):
    import os
    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    save_path = f'./output/{savedir}/'
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, img_path.split('\\')[-1])
    cv2.imwrite(save_name, img)
    return save_name
