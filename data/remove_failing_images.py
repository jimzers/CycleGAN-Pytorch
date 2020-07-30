import os
import glob
import PIL
from PIL import Image

file_arr = glob.glob('./chili-dog/*.jpg')

print(len(file_arr))

for filename in file_arr:
    try:
        Image.open(filename)
    except PIL.UnidentifiedImageError:
        print(filename)
        os.remove(filename)