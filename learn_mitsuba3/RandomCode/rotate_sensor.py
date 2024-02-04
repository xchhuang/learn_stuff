import mitsuba as mi
variants = mi.variants()
print('variants:', variants)
mi.set_variant(variants[0])
from mitsuba import ScalarTransform4f as T
import numpy as np
from tqdm import tqdm

scene_name = 'teapot'    # rover, teapot
spp = 16
max_depth = 2

scene = mi.load_file("../scenes/{:}/scene.xml".format(scene_name))


def main():
        
    for angle_y in tqdm([0, 35, 70, 105, 140, 175, 210, 245, 280, 315]):
        # camera_origin = T.rotate([0, 1, 0], 70).rotate([1, 0, 0], 30).translate([0.0, 23.0, 12.0]) @ mi.ScalarPoint3f(0, 0, 0)
        camera_origin = T.rotate([0, 1, 0], angle_y).rotate([1, 0, 0], 30).translate([0.0, 19.0, 10.0]) @ mi.ScalarPoint3f(0, 0, 0)

        # print('camera_origin:', camera_origin)

        sensor2 = mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            # 'to_world': T([
            #     [-0.00550949, -0.342144, -0.939631, 23.895],
            #     [1.07844e-005, 0.939646, -0.342149, 11.2207],
            #     [0.999985, -0.00189103, -0.00519335, 0.0400773],
            #     [0, 0, 0, 1],
            # ]),
            
            # 'to_world': mi.ScalarTransform4f.translate([0.0, 10.0, 20.0]) \
            #                                     .rotate([1, 0, 0], -22)   \
            #                                     .rotate([0, 1, 0], 20)   \
            #                                     .look_at(target=[0, 0, 0],
            #                                              origin=[0, 0, 0.01],
            #                                              up=[0, 1, 0]),

            'to_world': mi.ScalarTransform4f.look_at(target=[0, 2, 0], origin=camera_origin, up=[0, 1, 0]),

            'sampler': {
                'type': 'independent',
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': 256,
                'height': 256,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })

        integrator2 = mi.load_dict({
            'type': 'path',
            'max_depth': max_depth,
        })


        img = mi.render(scene, spp=spp, sensor=sensor2, integrator=integrator2)
        img = mi.Bitmap(img)

        # mi.Bitmap(img).write('{:}.ext'.format(scene_name))
        img = img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
        img.write('{:}_{:}.png'.format(scene_name, angle_y))




if __name__ == "__main__":
    main()

