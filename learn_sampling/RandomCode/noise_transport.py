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
    A = (grid_y[:-1, :-1], grid_x[:-1, :-1])
    # print(A[0].shape)
    B = (grid_y[:-1, 1:], grid_x[:-1, 1:])
    C = (grid_y[1:, 1:], grid_x[1:, 1:])
    D = (grid_y[1:, :-1], grid_x[1:, :-1])
    tri_ABC_y = torch.stack([A[0], B[0], C[0]], dim=2).reshape(-1, 3)
    tri_ABC_x = torch.stack([A[1], B[1], C[1]], dim=2).reshape(-1, 3)
    tri_ACD_y = torch.stack([A[0], C[0], D[0]], dim=2).reshape(-1, 3)
    tri_ACD_x = torch.stack([A[1], C[1], D[1]], dim=2).reshape(-1, 3)
    # triangle1_per_pixel = torch.cat([tri_ABC_x, tri_ABC_y], dim=0)
    # triangle2_per_pixel = torch.cat([tri_ACD_x, tri_ACD_y], dim=0)
    # triangles = torch.stack([triangle1_per_pixel, triangle2_per_pixel], dim=1)
    triangles_y = torch.cat([tri_ABC_y, tri_ACD_y], dim=0)
    triangles_x = torch.cat([tri_ABC_x, tri_ACD_x], dim=0)
    triangles = torch.stack([triangles_x, triangles_y], dim=-1)
    return triangles



# def triangles_from_grid(width, height):
#     # Generate mesh grids for both x and y coordinates
#     xv, yv = torch.meshgrid(torch.arange(width - 1), torch.arange(height - 1), indexing='ij')

#     # Calculate the total number of internal grid points
#     num_grid_points = (width - 1) * (height - 1)

#     # Initialize tensor to hold vertices for triangles
#     # For each grid point, there are 2 triangles, each with 3 vertices, and each vertex has 2 coordinates
#     triangles = torch.zeros((num_grid_points, 2, 3, 2), dtype=torch.float32).to(device)

#     # Calculate vertices for the first triangle (top-left, top-right, bottom-left)
#     triangles[:, 0, 0, :] = torch.stack([xv.flatten(), yv.flatten()], dim=1)                      # Top-left
#     triangles[:, 0, 1, :] = torch.stack([xv.flatten() + 1, yv.flatten()], dim=1)                  # Top-right
#     triangles[:, 0, 2, :] = torch.stack([xv.flatten(), yv.flatten() + 1], dim=1)                  # Bottom-left

#     # Calculate vertices for the second triangle (top-right, bottom-right, bottom-left)
#     triangles[:, 1, 0, :] = torch.stack([xv.flatten() + 1, yv.flatten()], dim=1)                  # Top-right
#     triangles[:, 1, 1, :] = torch.stack([xv.flatten() + 1, yv.flatten() + 1], dim=1)              # Bottom-right
#     triangles[:, 1, 2, :] = torch.stack([xv.flatten(), yv.flatten() + 1], dim=1)                  # Bottom-left

#     return triangles



def rasterize_triangles_batch(vertices, grid, width, height):
    batch_size = vertices.shape[0]
    # height, width = grid.shape[1], grid.shape[2]
    # grid = grid[0:height, 0:width, :]
    # Generate a grid of coordinates (height, width, 2)
    # y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    # grid = torch.stack((x, y), dim=-1).float().to(device)  # Shape: (height, width, 2)
    
    # Reshape for broadcasting to match (batch_size, height, width, 2)
    # grid = grid.reshape(1, height, width, 2).expand(batch_size, -1, -1, -1)
    
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
    # print('mask:', image.shape, mask.shape)

    image[mask] = 1.0  # This line was corrected
    
    return image



def create_a_circle(base_res):
    x_pos, y_pos = torch.meshgrid([torch.arange(0, base_res), torch.arange(0, base_res)])
    # x_pos = x_pos / base_res
    # y_pos = y_pos / base_res
    mgrid = torch.stack([x_pos, y_pos], dim=0)  # [C, D, H, W]
    mgrid = mgrid.float().to(device).view(2, -1).t()
    diff = mgrid - torch.tensor([[512, 400]]).to(device)
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

    vertices_batch = triangles_from_grid(create_mesh_grid(65, 65)).to(device)
    # print('vertices_batch:', vertices_batch.shape, vertices_batch.min(), vertices_batch.max())
    
    if False:
        small_batch = 64
        image_batch = 0
        num_triangles = 2
        iterations = int(vertices_batch.shape[0] // num_triangles)
        # image_batch = torch.zeros_like(mipmap[-1])   
        image_batch = []
        for i in tqdm(range(0, iterations, small_batch)):
            # print(i, i+small_batch)
            cur_vertices_batch1 = vertices_batch[i:i+small_batch]*16
            cur_vertices_batch2 = vertices_batch[i+iterations:i+iterations+small_batch]*16
            # print(i, i+small_batch, vertices_batch.shape[0], iterations)
            # print(i+iterations, i+iterations+small_batch, vertices_batch.shape[0], iterations)
            
            # print(cur_vertices_batch.shape, cur_vertices_batch.min(), cur_vertices_batch.max())
            cur_mask1 = rasterize_triangles_batch(cur_vertices_batch1, mesh_grids[0], base_res, base_res)
            cur_mask2 = rasterize_triangles_batch(cur_vertices_batch2, mesh_grids[0], base_res, base_res)
            # cur_mask_cat = torch.cat([cur_mask1, cur_mask2], dim=0).view(small_batch, 2, base_res, base_res)
            # print('cur_mask1:', cur_mask1.shape)
            # cur_mask = cur_mask1.bool() | cur_mask2.bool()
            cur_mask = torch.stack([cur_mask1, cur_mask2], dim=0)
            # print('cur_mask:', cur_mask.shape)
            interleaved = cur_mask.permute(1, 0, 2, 3).flatten(0, 1).view(small_batch, 2, base_res, base_res)
            interleaved = torch.any(interleaved, 1)
            # print('interleaved:', interleaved.shape)
            
            result = interleaved.float() * mipmap[0].unsqueeze(0)
            result = torch.sum(result, (1, 2)) / np.sqrt(16)
            # print('result:', result.shape)
            
            image_batch.append(result)
        
        image_batch = torch.stack(image_batch, dim=0).view(64, 64)

        if True:
            # image_batch_sum = torch.sum(image_batch, dim=0)
            plt.figure(1)
            plt.imshow(image_batch.detach().cpu().numpy(), cmap='gray')
            plt.show()

    # mipmap_base = mipmap[-1]
    # z = upsample_noise(mipmap_base.unsqueeze(0).unsqueeze(0), 16).squeeze(0).squeeze(0)
    z = mipmap[-1]
    if False:
        z = z.detach().cpu().numpy()
        plt.figure(1)
        plt.imshow(z, cmap='gray')
        plt.show()

    # target = z.unsqueeze(0)
    # modified_images = image_batch * target
    # print('modified_images:', modified_images.shape)
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(modified_images[0].detach().cpu().numpy(), cmap='gray')
    # plt.subplot(122)
    # plt.imshow(modified_images[1].detach().cpu().numpy(), cmap='gray')
    # plt.colorbar()
    # plt.show()
    # return
    

    # transport
    if True:
        cur_mesh_grid = mesh_grids[-1]
        next_mesh_grid = cur_mesh_grid.clone()  # must clone
        # res = cur_mesh_grid.shape[-1]
        # print('prev_mesh_grid:', prev_mesh_grid.shape, cur_mesh_grid.shape)
        
        plt.figure(1)
        for i in range(20):

            if True:
                plt.subplot(111)
                if i == 0:
                    plt.imshow(mipmap[-1].detach().cpu().numpy(), cmap='gray')
                else:
                    plt.imshow(next_mask.detach().cpu().numpy(), cmap='gray')
                # plt.subplot(122)
                # plt.imshow(mipmap[-1].detach().cpu().numpy(), cmap='gray')
                # plt.show()
                plt.pause(0.25)

            next_mesh_grid[:, :, 0:1] = next_mesh_grid[:, :, 0:1] - alpha
            next_mesh_grid_norm = (next_mesh_grid / 64 - 0.5) * 2.0
            next_mask = F.grid_sample(mipmap[-1].unsqueeze(0).unsqueeze(0), next_mesh_grid_norm.unsqueeze(0), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)

            if False:
                plt.figure(1)
                plt.scatter(next_mesh_grid[..., 0].cpu().numpy(), next_mesh_grid[..., 1].cpu().numpy(), s=1, c='r')
                plt.scatter(cur_mesh_grid[..., 0].cpu().numpy(), cur_mesh_grid[..., 1].cpu().numpy(), s=1, c='b')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.show()

            

        


    # linear interpolation
    if True:
        z = mipmap[-1].detach().cpu().numpy()
        plt.figure(1)
        for i in tqdm(range(2)):
            
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

