
import glob
import torchvision.transforms as T

from PIL import Image

preprocess = T.Compose([
   T.Resize(128),
   T.CenterCrop(128),
])

#x = preprocess(img)
files = glob.glob("D:\Simula_data\clean_optimized/**/*.jpg", recursive=True)
for path in files:
   img = preprocess(Image.open(path))
   name = path.split('\\', 1)
   imgs = name[1].split('\\',1)
   while(True):
      try:
         imgs = imgs[1].split('\\',1)
      except(IndexError):
         break

    ### MASKS should be mono either 255 or 0.

   img.save(f'D:/Simula_data/transformed_clean_opt/{imgs[0]}')