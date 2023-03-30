from numpy import load
from PIL import Image
import os

classcond = False

model_type = 'cropped_polyps_overtune2'
data = ('D:\Simula_data\Models\Temp\openai-2023-03-03-03-44-18-040654')
#data = ('Temp\openai-2023-01-30-22-43-22-233606')
print(data)
log_file = data + '\log.txt'


model_nr = 'NAN'
with open(log_file) as f:
    lines = f.readlines()
    model_nr = (lines[1].split('ema_0.9999_', 1)[1])
model_nr = (model_nr.split('.pt', 1)[0])
print(model_nr)

data = data +'\samples_1000x128x128x3.npz'


data = load(data)
print(data)


lst = data.files

if(classcond):
    imgs = data[lst[0]]

    obj = imgs.shape[0]
    for i in range(obj):
        img = imgs[i]
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img.save(f'generated_dataset/250_respace/img_class_cond_{i}_class_{data[lst[1]][i]}.png')
        #plt.figure()
        #plt.imshow(img, interpolation='none')
        #plt.show()
    #plt.show()

else:

    try: 
        os.mkdir(f"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/{model_type}/{model_nr}")
    except OSError as error: 
        print(error)  

    print(data[lst[0]].shape)
    for item in lst:
        print(data[item].shape)
        data = data[item]
        obj = data.shape[0]
        print(obj)
        for i in range(obj):
            img = data[i]
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img.save(f"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/{model_type}/{model_nr}/img_{i}.png")
            #img.save(f"C:/Users/Alexander-PC-hjemme/Desktop/UiO/Master_Thesis/guided-diffusion/{model_type}/{model_nr}/img_{i}.eps")
            #plt.figure()
            #plt.imshow(img, interpolation='none')
            #plt.show()
        #plt.show()