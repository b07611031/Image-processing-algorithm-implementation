import featureDetection as fdet
import featureDescriptor as fdes
import numpy as np
import math


def msop(img1, img2, nm='argmax'):
    # input images should be in RGB format.
    n = 2000
    imgs1, scaleMap1, ftx1, fty1 = fdet.multiscaleHarris(
        img1, n=n, des=True, nm=nm)
    featureVectors1 = fdes.descriptor(imgs1, scaleMap1, ftx1, fty1)
    imgs2, scaleMap2, ftx2, fty2 = fdet.multiscaleHarris(
        img2, n=n, des=True, nm=nm)
    featureVectors2 = fdes.descriptor(imgs2, scaleMap2, ftx2, fty2)

    dmin = math.dist(featureVectors1[0], featureVectors2[0])
    match_point = np.array([[0, 0, dmin]])
    for i in range(len(featureVectors1)):
        dmin = math.dist(featureVectors1[i], featureVectors2[0])
        matchpt = [i, 0, dmin]
        for j in range(len(featureVectors2)-1):
            e1 = math.dist(featureVectors1[i], featureVectors2[j])
            if e1 < dmin:
                dmin = e1
                matchpt = [i, j, dmin]
        # if abs(fty1[matchpt[0]]-fty2[matchpt[1]]) > 16:
        #     continue
        if matchpt[1] in match_point[:, 1]:
            for pt in np.argwhere(match_point[:, 1] == matchpt[1]):
                if matchpt[2] < match_point[pt[0], 2]:
                    match_point[pt[0]] = matchpt
        else:
            match_point = np.append(match_point, [matchpt], axis=0)
    # print(match_point.shape)
    match_point = match_point[:, :2].astype(int)
    hmp = []
    for mp in match_point:
        x1 = fty1[mp[0]]
        x2 = fty2[mp[1]] + img1.shape[1]
        y1 = ftx1[mp[0]]
        y2 = ftx2[mp[1]]
        if abs(y2-y1) < img1.shape[0]/16 and abs(x2-x1) < img1.shape[1]*0.8:
        # if abs(y2-y1) < img1.shape[0]/16:
            hmp.append([mp[0], mp[1]])
    hmp = np.array(hmp).astype(int)
    print('hmp', hmp.shape)
    k = 100000
    sigma = ftx1.max()/8
    pretotal = 0
    best_ptin = []
    import random
    for i in range(k):
        sample_index = random.sample(range(len(hmp)), 2)
        x1 = fty1[hmp[sample_index[0], 0]]
        x2 = fty1[hmp[sample_index[1], 0]]
        y1 = ftx1[hmp[sample_index[0], 0]]
        y2 = ftx1[hmp[sample_index[1], 0]]
        a = (y2 - y1) / (x2 - x1 + 1e-8)
        b = y1 - a * x1

        total_inlier = 0
        ptin = []
        for j, pt1 in enumerate(hmp):
            xj = fty1[pt1[0]]
            yj = ftx1[pt1[0]]
            y_estimate = a * xj + b
            if abs(y_estimate - yj) < sigma:
                ptin.append(pt1)
                total_inlier += 1

        if total_inlier > pretotal:
            pretotal = total_inlier
            best_ptin = np.array(ptin)
    # print(len(best_ptin))
    # return best_ptin.astype(int)
    match_point = np.array(best_ptin).astype(int)
    print('RANSAC:', match_point.shape)
    points1 = np.vstack((ftx1[match_point[:, 0]], fty1[match_point[:, 0]])).T
    points2 = np.vstack((ftx2[match_point[:, 1]], fty2[match_point[:, 1]])).T
    return points1, points2