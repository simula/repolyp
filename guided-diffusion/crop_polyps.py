import glob
import numpy as np

from PIL import Image


def cover_img(img,mask):
    val = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not(mask[i,j]):
                img[i,j] = [255]*3
            else:
                val.append(img[i,j])
    print(np.mean(val, axis=(0)))
    print(np.std(val, axis=(0)))
    return img

images = glob.glob("transformed_dataset_128/images/*.png")
masks = glob.glob("transformed_dataset_128/masks/*.png")
print(len(images))
print(len(masks))

for img, mask in zip(images, masks):
    path = img.split('//', 1)
    
    img = np.array(Image.open(img))
    mask = np.array(Image.open(mask))
    cropped_polyp = cover_img(img,mask)
    im = Image.fromarray(cropped_polyp)
    im.save(f"transformed_dataset_128/cropped/{path[1]}")