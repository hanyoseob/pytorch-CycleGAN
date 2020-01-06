import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

dir_result = './result/cyclegan/monet2photo/images'
lst_result = os.listdir(dir_result)

nx = 256
ny = 256
nch = 3

n = 8
m = 6
m_id = [0, 3, 1, 2]
n_id = 100

img = torch.zeros((n*(m-4), ny, nx, nch))

for i in range(m-4):
    for j in range(n):
        p = m_id[i + 0] + m * (j + n_id)
        # p = m_id[i + 2] + m*(j + n_id)
        q = n*i + j

        img[q, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_result[p]))[:, :, :nch])

        # print(lst_result[p])

img = img.permute((0, 3, 1, 2))

plt.figure(figsize=(n, (m-4)))
plt.axis("off")
# plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))

plt.show()

