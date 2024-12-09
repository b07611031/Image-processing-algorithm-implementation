def tonemapping(name, gamma=2.2):
    import numpy as np
    import cv2
    # hdrImg = cv2.imread(f'./output/{name}.hdr', flags=cv2.IMREAD_ANYDEPTH)
    hdrImg = cv2.cvtColor(cv2.imread(f'./output/{name}.hdr', flags=cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
    tonemap = cv2.createTonemap(gamma=gamma)
    res_ldr = tonemap.process(hdrImg)
    ldr_8bit = np.clip(res_ldr*255, 0, 255).astype('uint8')
    return ldr_8bit

    


# if __name__ == '__main__':
#     # import glob
#     # img_paths = glob.glob('.\\nikon/*.JPG')[1:]
#     # imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     #         for img_path in img_paths]
#     # from PIL import Image
#     # exposureTimes = [Image.open(img_path)._getexif()[33434] for img_path in img_paths]
    
#     hdr_path = './LDR-HDR-pair_Dataset-master/HDR/HDR_001.hdr'
#     hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)
#     print(hdr_img.shape)
#     print(type(hdr_img))
#     print(type(hdr_img[0,0,0]))
    
#     tonemap1 = cv2.createTonemap(gamma=2.2)
#     res_debevec = tonemap1.process(hdr_img.copy())
#     res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
#     cv2.imwrite("ldr_debevec.jpg", res_debevec_8bit)