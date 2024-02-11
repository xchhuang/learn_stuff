import mitsuba as mi
variants = mi.variants()
print('variants:', variants)
# mi.set_variant(variants[0])
mi.set_variant(variants[2])

from mitsuba import ScalarTransform4f as T

import drjit as dr
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# scene = mi.load_file('../scenes/cbox/scene.xml')
# scene = mi.load_file('../scenes/teapot/scene.xml')

scene = mi.load_dict(mi.cornell_box())




# Camera origin in world space
cam_origin = mi.Point3f(0, 0.9, 3)
# cam_origin = T.rotate([0, 1, 0], 30).rotate([1, 0, 0], 30).translate([0.0, 19.0, 10.0]) @ mi.ScalarPoint3f(0, 0, 0)


# Camera view direction in world space
target=[0, 2, 0]

cam_dir = dr.normalize(mi.Vector3f(0, -0.5, -1))
# cam_dir = dr.normalize(target - cam_origin)


# Camera width and height in world space
cam_width  = 2.0
cam_height = 2.0

# Image pixel resolution
image_res = [256, 256]
n_pixels = dr.prod(image_res)

# Construct a grid of 2D coordinates
x, y = dr.meshgrid(
    dr.linspace(mi.Float, -cam_width  / 2,   cam_width / 2, image_res[0]),
    dr.linspace(mi.Float, -cam_height / 2,  cam_height / 2, image_res[1])
)

# Ray origin in local coordinates
ray_origin_local = mi.Vector3f(x, y, 0)
# Ray origin in world coordinates
ray_origin = mi.Frame3f(cam_dir).to_world(ray_origin_local) + cam_origin


ray = mi.Ray3f(o=ray_origin, d=cam_dir)
# print(ray)
si = scene.ray_intersect(ray)

ambient_range = 0.75
ambient_ray_count = 16


# Initialize the random number generator
rng = mi.PCG32(size=n_pixels)


# sample_1, sample_2 = rng.next_float32(), rng.next_float32()
# print(sample_1[0:10], sample_1)
# sample_1[:] = np.random.rand(image_res[0]*image_res[1])
# sample_2[:] = np.random.rand(image_res[0]*image_res[1])
# sample_1, sample_2 = rng.next_float32(), rng.next_float32()
# print(sample_1[0:10])

# samples_1 = []
# samples_2 = []
samples_1 = dr.zeros(mi.Float, ambient_ray_count*image_res[0]*image_res[1])
samples_2 = dr.zeros(mi.Float, ambient_ray_count*image_res[0]*image_res[1])
for i in tqdm(range(ambient_ray_count)):
    samples_1[i*image_res[0]*image_res[1]:(i+1)*image_res[0]*image_res[1]] = np.random.rand(image_res[0]*image_res[1])
    samples_2[i*image_res[0]*image_res[1]:(i+1)*image_res[0]*image_res[1]] = np.random.rand(image_res[0]*image_res[1])



# Loop iteration counter
i = mi.UInt32(0)
j = dr.arange(mi.UInt32, 0, n_pixels)

# Accumulated result
result = mi.Float(0)

# Initialize the loop state (listing all variables that are modified inside the loop)
loop = mi.Loop(name="", state=lambda: (rng, i, result))


while loop(si.is_valid() & (i < ambient_ray_count)):
    
    # 1. Draw some random numbers
    # sample_1, sample_2 = rng.next_float32(), rng.next_float32()
    sample_1 = dr.gather(mi.Float, samples_1, i * n_pixels + j)
    sample_2 = dr.gather(mi.Float, samples_2, i * n_pixels + j)
    # 2. Compute directions on the hemisphere using the random numbers
    wo_local = mi.warp.square_to_uniform_hemisphere([sample_1, sample_2])

    # Alternatively, we could also sample a cosine-weighted hemisphere
    # wo_local = mi.warp.square_to_cosine_hemisphere([sample_1, sample_2])

    # 3. Transform the sampled directions to world space
    wo_world = si.sh_frame.to_world(wo_local)

    # 4. Spawn a new ray starting at the surface interactions
    ray_2 = si.spawn_ray(wo_world)

    # 5. Set a maximum intersection distance to only account for the close-by geometry
    ray_2.maxt = ambient_range

    # 6. Accumulate a value of 1 if not occluded (0 otherwise)
    result[~scene.ray_test(ray_2)] += 1.0
    # result[si.is_valid() & (i < ambient_ray_count)] += si.t
    # 7. Increase loop iteration counter
    i += 1

# Divide the result by the number of samples
result = result / ambient_ray_count


image = mi.TensorXf(result, shape=image_res)


import matplotlib.pyplot as plt

plt.figure(1)
plt.imshow(image, cmap='gray'); plt.axis('off')
plt.show()
