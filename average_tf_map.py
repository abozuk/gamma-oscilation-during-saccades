import numpy as np
import pylab as py


def average_tf(filename, x_shape=10, y_shape=100):
    data = np.load(filename)
    data = data[:4]

    # TODO do uogólnienia, jeśli wymiary by się nie dzieliły całkowicie
    f_shape = data.shape[3]//x_shape
    t_shape = data.shape[4]//y_shape
    average_data = np.zeros((data.shape[0], 19, 2,
                             f_shape,
                             t_shape))

    for x in range(0, f_shape):
        for y in range(0, t_shape):
            average_data[:, :, :, x, y] = np.mean(data[:,:,:,x:x+x_shape,y:y+y_shape], axis=(3,4))

    return average_data

if __name__ == "__main__":
    filename = 'output/data_matrix.npy'
    average = average_tf(filename)
    print(average.shape)
    avg1 = np.mean(average[:,:,0,:,:], axis=2)
    avg2 = np.mean(average[:, :, 1, :, :], axis=2)
    print(avg1 - avg2)

    #żeby sprawdzić czy mapki w miarę z sensem
    # py.imshow(average[0, 0, :, :], aspect='auto', origin='lower', interpolation='nearest')
    # py.show()
