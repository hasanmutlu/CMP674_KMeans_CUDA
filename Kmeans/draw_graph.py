import matplotlib.pyplot as plt



file = open("output.txt","r")
lines = file.readlines()
#data_count = int(lines[0])
#centroid_count = len(lines) - data_count - 1
i = 0
while i < len(lines):
    print(lines[i])
    c_x = int(lines[i].split(',')[0])
    c_y = int(lines[i].split(',')[1])
    x_arr = []
    y_arr = []
    count = int(lines[i].split(',')[2])
    for j in range (0,count):
        i = i + 1
        x = int(lines[i].split(',')[0])
        y = int(lines[i].split(',')[1])
        x_arr.append(x)
        y_arr.append(y)
    plt.plot(x_arr, y_arr, 'o')
    plt.plot(c_x, c_y, '^')
    i = i + 1

plt.show()