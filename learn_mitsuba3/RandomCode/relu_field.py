import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

import matplotlib.pyplot as plt


def plot_list(images, title=None):
    fig, axs = plt.subplots(1, len(images), figsize=(18, 3))
    for i in range(len(images)):
        axs[i].imshow(mi.util.convert_to_bitmap(images[i]))
        axs[i].axis('off')
    if title is not None:
        plt.suptitle(title)


# Rendering resolution
render_res = 256

# Number of stages
num_stages = 4

# Number of training iteration per stage
num_iterations_per_stage = 15

# learning rate
learning_rate = 0.2

# Initial grid resolution
grid_init_res = 16

# Spherical harmonic degree to be use for view-dependent appearance modeling
sh_degree = 2

# Enable ReLU in integrator
use_relu = True

# Number of sensors
sensor_count = 7

sensors = []

for i in range(sensor_count):
    angle = 360.0 / sensor_count * i
    sensors.append(mi.load_dict({
        'type': 'perspective',
        'fov': 45,
        'to_world': mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]) \
            .rotate([0, 1, 0], angle) \
            .look_at(target=[0, 0, 0],
                     origin=[0, 0, 1.3],
                     up=[0, 1, 0]),
        'film': {
            'type': 'hdrfilm',
            'width': render_res,
            'height': render_res,
            'filter': {'type': 'box'},
            'pixel_format': 'rgba'
        }
    }))

scene_ref = mi.load_file('../scenes/lego/scene.xml')
ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=64) for i in range(sensor_count)]
plot_list(ref_images, 'Reference images')
