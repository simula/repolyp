import glob
import torchvision.transforms as T

from PIL import Image

preprocess = T.Compose([
   T.Resize(128),
   T.CenterCrop(128),
])

files = glob.glob("Images_data/CVC-Clinic/**/*.png", recursive=True)
print(len(files))
for path in files:
   img = preprocess(Image.open(path))

   name = path.split('\\', 1)
   name = path.split('\\', 1)
   imgs = name[1].split('\\',1)
   imgs[1] =  imgs[1].split('.png', 1)[0]

    ### MASKS should be mono either 255 or 0.
   if(imgs[0]=='masks'):
      threshold = 128
      image_file = img.convert('L')
      # Threshold
      image_file = image_file.point( lambda p: 255 if p > threshold else 0 )
      img = image_file.convert('1')
   # save as png and not jpg to avoid compression for pixel values
   img.save(f'transformed_dataset_CVC/{imgs[0]}/img-{imgs[1]}.png')