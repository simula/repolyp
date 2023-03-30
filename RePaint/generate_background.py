import glob
import numpy as np
import torchvision.transforms as T

from PIL import Image

def cover_dataset():

    dataset = 'data/datasets/gts/over-tuned-cropped/*.png'
    imgs = glob.glob(dataset)
    print(len(imgs))

    for item in imgs:
        img =  np.array(Image.open(item))
        white_background = False
        if(white_background):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if(img[i,j,0] > 245 and img[i,j,1] > 245 and img[i,j,2] > 245):
                        img[i,j] = [0]*3
                    else:
                        img[i,j] = [255]*3
        else:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if(img[i,j,0] < 20 and img[i,j,1] < 20 and img[i,j,2] < 20):
                        img[i,j] = [0]*3
                    else:
                        img[i,j] = [255]*3
        print(item)
        file_name = item.split('data/datasets/gts/over-tuned-cropped\\', 1)[1]
        im = Image.fromarray(img)
        im.save(f"data/datasets/gt_keep_masks/over-tuned-cropped/{file_name}")

finale = cover_dataset()