import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mesh_grid(height, width):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    grid = torch.stack((x, y), dim=-1).float().to(device)  # Shape: (height, width, 2)
    return grid


def triangles_from_grid(grid):
    height, width = grid.shape[0], grid.shape[1]
    grid_x, grid_y = grid[..., 0], grid[..., 1]
    # Create mesh grid
    # y = torch.arange(height)
    # x = torch.arange(width)
    # grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate indices for vertices of two triangles (ABC and ACD) in each quad
    # Skip the last row and column to avoid out-of-bounds indexing
    A = (grid_y[:-1, :-1], grid_x[:-1, :-1])
    B = (grid_y[:-1, 1:], grid_x[:-1, 1:])
    C = (grid_y[1:, 1:], grid_x[1:, 1:])
    D = (grid_y[1:, :-1], grid_x[1:, :-1])
    
    # Combine indices to form triangles
    # Triangle ABC
    tri_ABC_y = torch.stack([A[0], B[0], C[0]], dim=2).reshape(-1, 3)
    tri_ABC_x = torch.stack([A[1], B[1], C[1]], dim=2).reshape(-1, 3)
    # Triangle ACD
    tri_ACD_y = torch.stack([A[0], C[0], D[0]], dim=2).reshape(-1, 3)
    tri_ACD_x = torch.stack([A[1], C[1], D[1]], dim=2).reshape(-1, 3)
    
    # Stack and reshape to get final vertices shape (number_of_triangles, 3, 2)
    triangles_y = torch.cat([tri_ABC_y, tri_ACD_y], dim=0)
    triangles_x = torch.cat([tri_ABC_x, tri_ACD_x], dim=0)
    triangles = torch.stack([triangles_x, triangles_y], dim=-1)
    
    return triangles



def rasterize_triangles_batch(vertices, width, height):
    batch_size = vertices.shape[0]
    
    # Generate a grid of coordinates (height, width, 2)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    grid = torch.stack((x, y), dim=-1).float().to(device)  # Shape: (height, width, 2)
    
    # Reshape for broadcasting to match (batch_size, height, width, 2)
    grid = grid.reshape(1, height, width, 2).expand(batch_size, -1, -1, -1)
    
    # Triangle vertices
    v0, v1, v2 = [vertices[:, i, :].unsqueeze(1).unsqueeze(1) for i in range(3)]
    
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    grid_v0 = grid - v0
    
    # Dot products for barycentric coordinates calculation
    dot00 = torch.sum(v0v2 * v0v2, dim=3)
    dot01 = torch.sum(v0v2 * v0v1, dim=3)
    dot02 = torch.sum(grid_v0 * v0v2, dim=3)
    dot11 = torch.sum(v0v1 * v0v1, dim=3)
    dot12 = torch.sum(grid_v0 * v0v1, dim=3)
    
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    w = 1 - u - v
    
    # Check if point is inside the triangle
    mask = (u >= 0) & (v >= 0) & (w >= 0)
    
    # Create the output image
    image = torch.zeros((batch_size, height, width)).to(device)
    # print('mask:', mask.shape)

    image[mask] = 1.0  # This line was corrected
    
    return image



def create_a_circle(base_res):
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



def upsample_noise(X, N):
    b, c, h, w = X.shape
    Z = torch.randn(b, c, N*h, N*w).to(X.device)
    Z_mean = Z.unfold(2, N, N).unfold(3, N, N).mean((4, 5))
    # print('Z_mean:', Z_mean.shape, X.shape)
    Z_mean = F.interpolate(Z_mean, scale_factor=N, mode='nearest')
    X = F.interpolate(X, scale_factor=N, mode='nearest')
    return X / N + Z - Z_mean


def main():

    alpha = 0.8
    mipmap_levels = [1, 2, 4, 8, 16]
    base_res = 1024

    mipmap = []
    high_res_map = create_a_circle(base_res)
    # high_res_map = torch.randn(base_res, base_res).float().to(device)
    # high_res_map = np.asarray(Image.open('../../../siren-pytorch/data/0.png').convert('L').resize((base_res, base_res))).astype(np.float32) / 255.0
    # high_res_map = torch.from_numpy(high_res_map).to(device)

    mesh_grids = []

    # plt.figure(1)
    for i in range(len(mipmap_levels)):
        x = F.interpolate(high_res_map.unsqueeze(0).unsqueeze(0), scale_factor=1/mipmap_levels[i], mode='bilinear').squeeze(0).squeeze(0)
        mipmap.append(x)

        mesh_grids.append(create_mesh_grid(x.shape[0], x.shape[1]))
    #     plt.subplot(1, len(mipmap_levels), i+1)
    #     plt.imshow(x.cpu().numpy(), cmap='gray')
    # plt.show()
        
    
    # vertices_batch = torch.tensor([
    #     [[10.0, 10.0], [50.0, 10.0], [30.0, 40.0]],
    #     [[210.0, 210.0], [250.0, 210.0], [230.0, 240.0]]
    # ]).to(device)   # Shape: (num_triangles, num_triangle_vertices=3, xy_coordinates=2)

    vertices_batch = triangles_from_grid(mesh_grids[0])
    # print('vertices_batch:', vertices_batch.shape, vertices_batch.min(), vertices_batch.max())
    # return
    small_batch = 9
    for i in range(0, vertices_batch.shape[0], small_batch):
        cur_vertices_batch = vertices_batch[i:i+small_batch]
        image_batch = rasterize_triangles_batch(cur_vertices_batch, base_res, base_res)
        # print(image_batch.shape, cur_vertices_batch.shape)
        if True:
            image_batch_sum = torch.sum(image_batch, dim=0)
            plt.figure(1)
            plt.imshow(image_batch_sum.detach().cpu().numpy(), cmap='gray')
            plt.show()

    mipmap_base = mipmap[-1]
    z = upsample_noise(mipmap_base.unsqueeze(0).unsqueeze(0), 16).squeeze(0).squeeze(0)

    if False:
        z = z.detach().cpu().numpy()
        plt.figure(1)
        plt.imshow(z, cmap='gray')
        plt.show()

    target = z.unsqueeze(0)
    modified_images = image_batch * target
    print('modified_images:', modified_images.shape)
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(modified_images[0].detach().cpu().numpy(), cmap='gray')
    plt.subplot(122)
    plt.imshow(modified_images[1].detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.show()
    return
    
    plt.figure(1)
    for i in tqdm(range(20)):
        
        plt.imshow(z, cmap='gray')
        
        x_m1 = np.pad(z,((0,0),(1,0)), mode='constant')[:, :-1]    # motion
        
        z = alpha * x_m1 + (1 - alpha) * z
        
        plt.pause(0.25)
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(x, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(z, cmap='gray')
    # plt.show()


if __name__ == "__main__":
    main()

