import os

import glob
import sys
import numpy as np

from PIL import Image

def cover_dataset():

    polyp_folder = 'polyps-1\\'

    dataset = f'D:\Simula_data\Images_data\cropped\{polyp_folder}\*.png'
    main_folder = 'D:\Simula_data\Images_data\\'
    #folder1 = 'images\\'
    folder2 = 'masks\\'
    imgs = glob.glob(dataset)

    for item in imgs:
        file_name = item.split(f'D:\Simula_data\Images_data\cropped\{polyp_folder}', 1)[1]

        
        path = main_folder+folder2+polyp_folder
        try: 
            os.mkdir(path) 
        except OSError as error:
            pass 
            #print('\nFolder already made.')

        img =  np.array(Image.open(main_folder+folder2+file_name))
        im = Image.fromarray(img) 
        #print(main_folder+folder1+polyp_folder+file_name)
        im.save(path+file_name)
        sys.exit()


finale = cover_dataset()