import utils.alignment as align
import utils.ldr2hdr as hdr
import utils.tonemapping as tp
# from utils.alignment import ward_alignment
# from utils.ldr2hdr import debevec_hdr
# from utils.tonemapping import tonemapping

import cv2
from PIL import Image
import argparse
import glob


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', type=str, default='./vending',
                        help='If you have a set of LDR images that you want to transform into an HDR image, \
                            the input images should be in *.JPG, *jpg, or *.jpeg format and should contain exposure time information in their metadata.')
    parser.add_argument('--savename', default='exp', type=str,
                        help='save to output/name.hdr, output/name.jpg')
    parser.add_argument('--alignment', type=str, default=False,
                        help='Alignment was implemented in the MTB algorithm. \
                            Use the —alignment flag to determine whether to apply the MTB. By default, it is set to False.')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    img_paths = glob.glob(f'{opt.dir_path}/*.JPG')
    img_paths += glob.glob(f'{opt.dir_path}/*.jpeg')
    imgs, exposureTimes = [], []
    for img_path in img_paths:
        imgs.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        exposureTimes.append(Image.open(img_path)._getexif()[33434])

    if bool(opt.alignment) == True:
        align_paths = align.ward_alignment(imgs, img_paths=img_paths,
                                           max_offset=8, thr=4, gray='grey')
        imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                for img_path in align_paths]

    hdrImg = hdr.debevec_hdr(imgs, exposureTimes, n=500, name=opt.savename)
    cv2.imwrite(f'./output/{opt.savename}.hdr', hdrImg)
    print(
        f'The recovered HDR image has been saved to ./output/{opt.savename}.hdr')
    ldr_8bit = tp.tonemapping(opt.savename, gamma=2.2)
    cv2.imwrite(f"./output/{opt.savename}.jpg", ldr_8bit)
    cv2.imwrite(f"./output/{opt.savename}.png", ldr_8bit)
    print(
        f"The tone-mapped image has been saved to ./output/{opt.savename}.jpg")
