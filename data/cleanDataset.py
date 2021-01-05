from os import listdir
from os.path import isfile, join

files = [f for f in listdir("C:/Users/andyb/PycharmProjects/resnetRopa/images") if isfile(join("C:/Users/andyb/PycharmProjects/resnetRopa/images", f))]

f = open("styles.txt")
o = open("cleanStyles.txt", "w")

i = 0
linea = f.readline()
while linea != "":
	if linea.split("/")[1].split("\t")[0] in files:
		o.write(linea)
	linea = f.readline()
	i += 1
	print(i)
	