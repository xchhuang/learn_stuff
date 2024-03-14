import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mesh_grid(base_res):
    x_pos, y_pos = torch.meshgrid([torch.arange(0, base_res), torch.arange(0, base_res)])
    # x_pos = x_pos / base_res
    # y_pos = y_pos / base_res
    mgrid = torch.stack([x_pos, y_pos], dim=0)  # [C, D, H, W]
    mgrid = mgrid.float().to(device).view(2, -1).t()
    diff = mgrid - torch.tensor([[512, 512]]).to(device)
    # print('diff:', diff.shape)
    dist2 = torch.sum(diff ** 2, -1)
    idx = dist2 < 40000
    dist2[idx] = 1.0
    dist2[~idx] = 0
    dist2 = dist2.view(base_res, base_res)
    # print('mgrid:', mgrid.shape, dist2.shape)
    # plt.figure(1)
    # plt.imshow(dist2.detach().cpu().numpy(), cmap='gray')
    # plt.show()
    return dist2



def main():

    alpha = 0.8
    mipmap_levels = [1, 2, 4, 8, 16]
    base_res = 1024

    mipmap = []
    high_res_map = create_mesh_grid(base_res)
    # high_res_map = torch.randn(base_res, base_res).float().to(device)
    # high_res_map = np.asarray(Image.open('../../../siren-pytorch/data/0.png').convert('L').resize((base_res, base_res))).astype(np.float32) / 255.0
    # high_res_map = torch.from_numpy(high_res_map).to(device)

    mesh_grids = []

    # plt.figure(1)
    for i in range(len(mipmap_levels)):
        x = F.interpolate(high_res_map.unsqueeze(0).unsqueeze(0), scale_factor=1/mipmap_levels[i], mode='bilinear').squeeze(0).squeeze(0)
        mipmap.append(x)

        mesh_grids.append(np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1])))
    #     plt.subplot(1, len(mipmap_levels), i+1)
    #     plt.imshow(x.cpu().numpy(), cmap='gray')
    # plt.show()
        
    
    plt.figure(1)
    z = mipmap[-1].detach().cpu().numpy()
    for i in tqdm(range(10)):
        
        plt.imshow(z, cmap='gray')
        
        x_m1 = np.pad(z,((0,0),(1,0)), mode='constant')[:, :-1]    # motion
        
        z = alpha * x_m1 + (1 - alpha) * z
        
        plt.pause(0.5)
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(x, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(z, cmap='gray')
    # plt.show()


if __name__ == "__main__":
    main()

