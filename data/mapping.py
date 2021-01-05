import os
f = open("cleanStyles.txt")
s = set()

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

line = f.readline()
while line != "":
    label = line.split("\t")[1].split("\n")[0]
    s.add(label)
    line = f.readline()

f.close()

m = open("mapping.txt", "w")
i = 0
for label in s:
	m.write(label + "\t" + str(i) + "\n")
	i += 1
	
print(len(s))
	
m.close()
