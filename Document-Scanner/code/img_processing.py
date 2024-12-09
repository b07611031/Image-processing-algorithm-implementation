import cv2 as cv
import numpy as np
import os


def save_output(img, img_path, suffix=''):
    img[img > 255] = 255
    img[img < 0] = 0
    os.makedirs('./output', exist_ok=True)
    file_name = img_path.split('/')[-1].split('.')
    save_path = './output/' + file_name[0] + \
        '_' + suffix + '.' + file_name[-1]
    cv.imwrite(save_path, img)
    return save_path


def aug(img, img_path, brightness=0, contrast=0):
    output_img = (img-255/2) * pow(2, contrast/4) + 255/2 + brightness
    output_path = save_output(output_img, img_path, 'aug')
    return output_path


def rotate(img, img_path):
    output_img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    output_path = save_output(output_img, img_path, 'rotate')
    return output_img, output_path


def scan(img_path, func=''):
    output_img = cv.imread(img_path)
    hsv = cv.cvtColor(output_img, cv.COLOR_BGR2HSV)
    guassian = cv.GaussianBlur(hsv[:, :, 1], (11, 11), 0)
    ret, thresh = cv.threshold(
        guassian, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    contours, _ = cv.findContours(
        thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    cnt = contours[0]
    hull = cv.convexHull(cnt)
    # transform contour to rectangle img
    epsilon = 0.1*cv.arcLength(hull, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    if func == 'find':
        cv.drawContours(output_img, [approx], 0, (0, 255, 0), 3)
        output_path = save_output(output_img, img_path, 'find')
        return output_path
    # the document size
    x, y, w, h = cv.boundingRect(cnt)
    src = np.float32(np.reshape(approx, (4, -1)))
    dst = np.float32([[w, 0], [0, 0], [0, h], [w, h]])
    m = cv.getPerspectiveTransform(src, dst)
    doc = cv.warpPerspective(output_img, m, (w, h))
    if func == 'binarize':
        # turn document img to gray and then binarize
        doc_gray = cv.cvtColor(doc, cv.COLOR_BGR2GRAY)
        _, doc = cv.threshold(
            doc_gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # print(doc.shape)
    output_path = save_output(doc, img_path, 'result')
    return output_path


# def main():
#     img = cv.imread('img/invoice.jpg')


# if __name__ == "__main__":
#     main()
