import matplotlib.pyplot as plt
data_x_arr = []
data_y_arr = []
cent_x_arr = []
cent_y_arr = []
file = open("output.txt","r")
lines = file.readlines()
data_count = int(lines[0])
centroid_count = len(lines) - data_count - 1
for i in range(1,len(lines)):
    x = int(lines[i].split(',')[0])
    y = int(lines[i].split(',')[1])
    if i <= data_count:
        data_x_arr.append(x)
        data_y_arr.append(y)
    else:
        cent_x_arr.append(x)
        cent_y_arr.append(y)




print(centroid_count)
plt.plot(data_x_arr, data_y_arr, 'o')
plt.plot(cent_x_arr, cent_y_arr,'o')
plt.show()