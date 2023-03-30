import cv2
import glob
import itertools

i = 0
for x in range(100):
    model_nr = '060000'
    #in_img = f"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/generated_dataset/{model_nr}_32_flip/img_{x}.png"
    in_img = f"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/generated_dataset/{model_nr}/img_{x}.png"
    template = cv2.imread(in_img, 0)
    template_horz = cv2.flip(template,1)
    #files = glob.glob("D:/Simula_data/Images_data/cropped/polyps-1/*.png")
    files = glob.glob(f"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/transformed_dataset_128/cropped/*.png")

    max_conf = 0
    out_name = ""
    name_score = {}
    #print((f'\"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/generated_dataset/{model_nr}_32_flip/img_{x}.png\"'))
    print((f'\"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/generated_dataset/{model_nr}/img_{x}.png\"'))
    
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
    N = 5
    top_N = dict(itertools.islice(sorted_x.items(),N))
    keysList = list(top_N.keys())

    if(max_conf > 0.97):
        i+=1
    """
    # Print only orginal name of imgs
    ch = 'clean_'
    for item in keysList:
        listOfWords = item.split(ch, 1)
        print(listOfWords[1])
    print(keysList)"""
    print(f'File number: {x}')
    print(max_conf)
    print(out_name)
    #print(top_N)
    print('-------------------------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------------------------')
    #print(f"C:\Users\\'Alexander-PC-hjemme'\Desktop/UiO/Master_Thesis/guided-diffusion/generated_dataset/060000_32/img_{x}.png\n")
    #print('-------------------------------------------------------------------------------------------------------------')
print(f'Total imgs with over 0.97 is: {i}')