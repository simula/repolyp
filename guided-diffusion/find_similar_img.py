import cv2
import glob
import itertools
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()

### USES cv2.matchTemplate GRAYSCALE IMAGE AND/OR CALCULATE DISTANCE IS L2 DISTANCE BETWEEN RGB IMAGES.
i=0
list_of_conf = []
def calculateDistance(i1, i2):
    #print(np.sum((i1-i2)**2))
    diff = i1 - i2


    return np.sqrt(np.sum((diff/(16384*3))**2))


for x in range(1000):
    model_nr = '020000'
    folder_ = 'cropped_polyps_overtune'
    #model_nr = 'inpainted'
    #folder_ = 'C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/RePaint/log/inpaint_background_clean'
    in_img = f"{folder_}/{model_nr}/img_{x}.png"
    #template = cv2.imread(in_img, 0)
    #template_horz = cv2.flip(template,1)

    #print(in_img)
    template = cv2.imread(in_img, 0)
    template_horz = cv2.flip(template,1)


    template_color_rgb = cv2.imread(in_img, 1)
    template_horz_rgb = cv2.flip(template_color_rgb,1)
    
    #files = glob.glob("unlabeled-dataset-128x128/train/unlabeled_128/*.jpg")
    images = [cv2.imread(image,0) for image in glob.glob("transformed_dataset_128/cropped/*.png")]

    images_rgb = [cv2.imread(image,1) for image in glob.glob("transformed_dataset_128/cropped/*.png")]
    names = [image for image in glob.glob("transformed_dataset_128/cropped/*.png")]

    #images_rgb = [cv2.imread(image,1) for image in glob.glob(f"{folder_}//{model_nr}/*.png")] #remove
    #names = [image for image in glob.glob(f"{folder_}//{model_nr}/*.png")]#remove

    max_conf = 0
    out_name = ""
    out_name2 =""
    name_score = {}
    name_score2 = {}
    l2_dis = 1
    for name,img_rgb,img in zip(names,images_rgb,images):
        name_score[name] = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max()
        name_score2[name] = calculateDistance(template_color_rgb,img_rgb)

        if (cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max() > max_conf):
            max_conf = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max()
            out_name = name
        if (cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max() > max_conf):
            max_conf = cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max()
            out_name = name
            name_score[name] = cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max()
        
        if (l2_dis > calculateDistance(template_color_rgb,img_rgb)):
            name_score2[name] = calculateDistance(template_color_rgb,img_rgb)
            l2_dis = calculateDistance(template_color_rgb,img_rgb)
            out_name2 = name
        if (l2_dis > calculateDistance(template_horz_rgb,img_rgb)):
            name_score2[name] = calculateDistance(template_horz_rgb,img_rgb)
            l2_dis = calculateDistance(template_horz_rgb,img_rgb)
            out_name2 = name


    #print(l2_dis)
    if((max_conf < 0.95) or (max_conf==1.0)):
        continue
        

    
    #print(max_conf)
    #print(len(name_score))
    #print(len(name_score2))
    sorted_x = dict(sorted(name_score.items(), key=lambda item: item[1]))
    sorted_x = dict(reversed(sorted_x.items()))
    cv2_dist = dict(itertools.islice(sorted_x.items(),5))
    keysList = list(cv2_dist.keys())

    sorted_x = dict(sorted(name_score2.items(), key=lambda item: item[1]))
    #sorted_x = dict(reversed(sorted_x.items()))
    l2_dist = dict(itertools.islice(sorted_x.items(),10))
    keysList = list(l2_dist.keys())


    """
    # Print only orginal name of imgs
    ch = 'clean_'
    for item in keysList:
        listOfWords = item.split(ch, 1)
        print(listOfWords[1])
    print(keysList)"""
    print('---------------')
    #print(max_conf)
    #print(out_name)
    #print(out_name2)
    print(cv2_dist)
    print()
    #print(l2_dist)
    print(f'\nFile number: {x}')
    print(f'Refrence image: {folder_}/{model_nr}/img_{x}.png')
    i+=1
print(f'Total number of images memorized or close to being are: {i}')
print("--- %s seconds ---" % (time.time() - start_time))
plt.hist(list_of_conf, bins=[0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95])
plt.show()


sys.exit()

for x in range(100):
    model_nr = '150000'
    in_img = f"generated_fine-tuned//{model_nr}/img_{x}.png"
    template = cv2.imread(in_img, 0)
    template_horz = cv2.flip(template,1)
    files = glob.glob("unlabeled-dataset-128x128/train/unlabeled_128/*.jpg")

    max_conf = 0
    out_name = ""
    name_score = {}
    for name in files:
        img = cv2.imread(name, 0)
        name_score[name] = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max()
        if (cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max() > max_conf):
            max_conf = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max()
            out_name = name
        if (cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max() > max_conf):
            max_conf = cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max()
            out_name = name
            name_score[name] = cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max()

    sorted_x = dict(sorted(name_score.items(), key=lambda item: item[1]))
    sorted_x = dict(reversed(sorted_x.items()))
    cv2_dist = dict(itertools.islice(sorted_x.items(),5))
    keysList = list(cv2_dist.keys())
    """
    # Print only orginal name of imgs
    ch = 'clean_'
    for item in keysList:
        listOfWords = item.split(ch, 1)
        print(listOfWords[1])
    print(keysList)"""
    print('---------------')
    print(max_conf)
    print(out_name)
    print(cv2_dist)
    print(f'File number: {x}')
    print(f'generated_dataset/{model_nr}/img_{x}.png')


i=0
list_of_conf = []
def calculateDistance(i1, i2):
    #print(np.sum((i1-i2)**2))
    diff = i1 - i2
    return np.sqrt(np.sum(diff ** 2))


for x in range(1000):
    model_nr = '026000'
    folder_ = 'generated-fine-tuned-best-pre-0.3-dropout'
    in_img = f"{folder_}//{model_nr}/img_{x}.png"
    template = cv2.imread(in_img, 0)
    template_horz = cv2.flip(template,1)

    template_color_rgb = cv2.imread(in_img, 1)
    template_horz_rgb = cv2.flip(template_color_rgb,1)
    
    #files = glob.glob("unlabeled-dataset-128x128/train/unlabeled_128/*.jpg")
    images = [cv2.imread(image,0) for image in glob.glob("transformed_dataset_128/images/*.png")]
    images_rgb = [cv2.imread(image,1) for image in glob.glob("transformed_dataset_128/images/*.png")]
    #images = [cv2.imread(image,0) for image in glob.glob("unlabeled-dataset-128x128/train/masked_images/*.png")]
    names = [image for image in glob.glob("transformed_dataset_128/images/*.png")]
    #names = [image for image in glob.glob("unlabeled-dataset-128x128/train/masked_images/*.png")]

    max_conf = 0
    out_name = ""
    out_name2 =""
    name_score = {}
    name_score2 = {}
    l2_dis = 0
    for img,name,img_rgb in zip(images,names,images_rgb):
        name_score[name] = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max()
        name_score2[name] = calculateDistance(template_color_rgb,img_rgb)

        if (cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max() > max_conf):
            max_conf = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED).max()
            out_name = name
        if (cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max() > max_conf):
            max_conf = cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max()
            out_name = name
            name_score[name] = cv2.matchTemplate(img, template_horz, cv2.TM_CCOEFF_NORMED).max()
        
        if (l2_dis==0 or l2_dis > calculateDistance(template_color_rgb,img_rgb)):
            name_score2[name] = calculateDistance(template_color_rgb,img_rgb)
            l2_dis = calculateDistance(template_color_rgb,img_rgb)
            out_name2 = name
        if (l2_dis==0 or l2_dis > calculateDistance(template_horz_rgb,img_rgb)):
            name_score2[name] = calculateDistance(template_horz_rgb,img_rgb)
            l2_dis = calculateDistance(template_horz_rgb,img_rgb)
            out_name2 = name

    """if (max_conf != 1):
        #print(x)
        list_of_conf.append(max_conf)
    if(max_conf < 0.92):
        continue"""
        

    print(len(name_score))
    print(len(name_score2))
    sorted_x = dict(sorted(name_score.items(), key=lambda item: item[1]))
    sorted_x = dict(reversed(sorted_x.items()))
    cv2_dist = dict(itertools.islice(sorted_x.items(),5))
    keysList = list(cv2_dist.keys())

    sorted_x = dict(sorted(name_score2.items(), key=lambda item: item[1]))
    sorted_x = dict((sorted_x.items()))
    l2_dist = dict(itertools.islice(sorted_x.items(),5))
    keysList = list(l2_dist.keys())
    """
    # Print only orginal name of imgs
    ch = 'clean_'
    for item in keysList:
        listOfWords = item.split(ch, 1)
        print(listOfWords[1])
    print(keysList)"""
    print('---------------')
    #print(max_conf)
    #print(out_name)
    print(out_name2)
    #print(cv2_dist)
    print(l2_dist)
    print(f'File number: {x}')
    print(f'{folder_}/{model_nr}/img_{x}.png')
    i+=1
    input()
print(f'Total number of images memorized or close to being are: {i}')
print("--- %s seconds ---" % (time.time() - start_time))
plt.hist(list_of_conf, bins=[0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95])
plt.show()
