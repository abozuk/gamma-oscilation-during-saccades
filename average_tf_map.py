import numpy as np
import pylab as py


def average_tf(filename):
    data = np.load(filename)
    average_data = np.zeros((19, 2, 60, 500))
    for i in range(0, data.shape[2]):
        for j in range(0, data.shape[0]):
            average_data[:, i, :, :] += data[j, :, i, :, :]
        print(average_data)
        average_data = average_data / data.shape[0]
    return average_data


filename = 'data_matrix.npy'
average = average_tf(filename)

#żeby sprawdzić czy mapki w miarę z sensem
py.imshow(average[0, 0, :, :], aspect='auto', origin='lower', interpolation='nearest')
py.show()
