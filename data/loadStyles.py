f = open("styles.csv", "r");
o = open("styles.txt", "w");

line = f.readline();
while line != "":
	sline = line.split(",")
	o.write("images/" + sline[0] + ".jpg" + "\t" + sline[4] + "\n")
	line = f.readline()
	
f.close()
o.close()
	