import torch as t
import numpy as np

def img2points_group(img, window_size):
    batch,height,width,channel = img.shape[0],img.shape[2],img.shape[3],img.shape[1]
    img = img.reshape(channel,height,width).transpose(1,2,0)
    img = np.pad(img, ((window_size // 2,window_size // 2),(window_size // 2,window_size // 2),(0,0)),'edge')
    x = t.Tensor(img).cuda()
    roll_list = []
    for i in range(-(window_size // 2),window_size // 2+1):
        for j in range(-(window_size // 2),window_size // 2+1):
            roll_coord = (-i,-j)
            roll_list.append(roll_coord)
    y = t.roll(x,shifts=roll_list[0], dims=(0,1))
    data_all = y[:,:,:,None]
    for i in range(1, window_size*window_size):
        y = t.roll(x,shifts=roll_list[i], dims=(0,1))[:,:,:,None]
        data_all = t.cat((data_all,y),3)
    data_all = data_all[window_size // 2:-(window_size // 2),window_size // 2:-(window_size // 2),:,:]
    data_all = data_all.reshape(data_all.shape[0],data_all.shape[1],data_all.shape[2],window_size,window_size)

    return data_all.cpu()