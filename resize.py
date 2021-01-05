from PIL import Image
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir("bigImages") if isfile(join("bigImages", f))]
size = 390, 520

i = 0;
for f in onlyfiles:
	im = Image.open("bigImages/" + f)
	im_resized = im.resize(size, Image.ANTIALIAS)
	im_resized.save("resized/" + f , "JPEG")
	i += 1
	print(i)