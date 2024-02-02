import mitsuba as mi
variants = mi.variants()
print('variants:', variants)
mi.set_variant(variants[0])
from mitsuba import ScalarTransform4f as T
import numpy as np


scene_name = 'teapot'    # rover, teapot
spp = 64
max_depth = 2

scene = mi.load_file("../scenes/{:}/scene.xml".format(scene_name))


sensor2 = mi.load_dict({
    'type': 'perspective',
    # 'fov': 39.3077,
    'to_world': T([
        [-0.00550949, -0.342144, -0.939631, 23.895],
        [1.07844e-005, 0.939646, -0.342149, 11.2207],
        [0.999985, -0.00189103, -0.00519335, 0.0400773],
        [0, 0, 0, 1],
    ]),
    'sampler': {
        'type': 'independent',
        'sample_count': spp
    },
    # 'film': {
    #     'type': 'hdrfilm',
    #     'width': 256,
    #     'height': 256,
    #     'rfilter': {
    #         'type': 'tent',
    #     },
    #     'pixel_format': 'rgb',
    # },
})

integrator2 = mi.load_dict({
    'type': 'path',
    'max_depth': max_depth,
})


img = mi.render(scene, spp=spp, sensor=sensor2, integrator=integrator2)

mi.Bitmap(img).write('{:}.exr'.format(scene_name))

