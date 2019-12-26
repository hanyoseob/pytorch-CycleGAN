import os
import numpy as np
import matplotlib.pyplot as plt

# fig, (splt1, splt2) = plt.subplots(1, 2)

sz = 256

str_dataset = {'train', 'val', 'test'}

for phs in str_dataset:

    dir_data = './data/facades/%s/' % phs
    lst_data = os.listdir(dir_data)

    lst_data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for i_, lst_ in enumerate(lst_data):
        data_ = plt.imread(os.path.join(dir_data, lst_))

        label_ = data_[:, :sz, :]
        input_ = data_[:, sz:, :]

        np.save(os.path.join(dir_data, "label_%05d.npy" % i_), np.float32(label_))
        np.save(os.path.join(dir_data, "input_%05d.npy" % i_), np.float32(input_))
