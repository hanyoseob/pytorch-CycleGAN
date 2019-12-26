import os
import numpy as np
import matplotlib.pyplot as plt

## setup scope title
# scope = 'trans2dens_1linear_l2'
scope = 'trans2dens_1linear_l1'

dir_output = os.path.join('./result', scope)
dir_data = './data/'

num_train = 8000
num_val = 1000
num_test = 1000

## load list of output
lst_output = os.listdir(dir_output)
lst_output.sort(key=lambda l: int(''.join(filter(str.isdigit, l))))

fig, ax = plt.subplots()

for i, name in enumerate(lst_output, num_train + num_val):
    output = np.load(os.path.join(dir_output, name))
    label = np.load(os.path.join(dir_data, 'label_%05d_1d.npy' % i))

    ax.plot(output, 'b-')
    ax.plot(label, 'r--')

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 40)

    ax.grid(True)
    ax.set_ylabel('Density')
    ax.set_xlabel('Radius')
    ax.legend(('Output', 'Label'), loc='upper right')
    plt.title('%04d / %04d' % (i - (num_train + num_val) + 1, num_test))

    plt.pause(0.3)
    plt.cla()
