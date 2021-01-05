f = open("cleanStyles.txt")
m = open("mapping.txt")
t = open("trainvalID.txt", "w")

dict = {}
for item in m:
	dict[item.split("\t")[0]] = item.split("\t")[1].split("\n")[0]
	

line = f.readline()
while(line != ""):
	label = line.split("\t")[1].split("\n")[0]
	t.write(line.split("\t")[0] + "\t" + dict[label] + "\n")
	line = f.readline()
	
f.close()
m.close()
t.close()