import random

arr = []
for i in range (0, 36000):
	arr.append(1)
	

f = open("trainvalID.txt")
t = open("train.txt", "w")
v = open("test.txt", "w")

resto = sum(1 for line in f) - 36000
f.close()
f = open("trainvalID.txt")

print(resto)

for i in range (0, resto):
	arr.append(0)
	
random.shuffle(arr)	
	
	
for x in arr:
	line = f.readline()
	if x == 1:
		t.write(line)
	else:
		v.write(line)
		
f.close()
t.close()
v.close()
		