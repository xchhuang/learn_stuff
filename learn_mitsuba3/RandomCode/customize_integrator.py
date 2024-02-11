import mitsuba as mi
variants = mi.variants()
print('variants:', variants)
mi.set_variant(variants[2])
from mitsuba import ScalarTransform4f as T
import numpy as np
from tqdm import tqdm
import os
import platform
import drjit as dr

# print(platform.system())

scene_name = 'teapot'    # rover, teapot
scene = mi.load_file("../scenes/{:}/scene.xml".format(scene_name))


if platform.system() == 'Windows':
    spp = 64
    max_depth = 2
    res = 512
    angle_y_list = [0, 1, 2]
elif platform.system() in ['Darwin']:
    spp = 16
    max_depth = 2
    res = 256
    angle_y_list = [0]
else:
    raise ValueError('Unknown platform')


output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)




class MyIntegrator(mi.SamplingIntegrator):
    def __init__(self, props = mi.Properties()):
        super().__init__(props)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
        ) -> tuple[mi.Color3f, bool, list[float]]:
        
        L = mi.Spectrum(0)
        ray = mi.Ray3f(ray)
        active = mi.Bool(active)

        pi: mi.PreliminaryIntersection3f = scene.ray_intersect_preliminary(ray)

        # As an example, only interested in the first shape listed in the scene
        # my_active = active & dr.eq(pi.shape, sh[0]) 
        
        # How many rays hit this shape
        # print(dr.width(dr.compress(my_active)))
        
        return L, mi.Bool(True), []
    
    




def main():
    
            

    mi.register_integrator("MyIntegrator", lambda props: MyIntegrator(props))

    # angle_y_list = np.arange(0, 360, 30)
    # angle_y_list = [0, 1, 2]
    print('angle_y_list:', angle_y_list)

    for seed, angle_y in enumerate(tqdm(angle_y_list)):
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
                'type': 'stratified',
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': res,
                'height': res,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })

        
        integrator2 = mi.load_dict({
            # 'type': 'path',
            'type': 'MyIntegrator',
            # 'max_depth': max_depth,
        })


        img = mi.render(scene, spp=spp, sensor=sensor2, integrator=integrator2, seed=seed) # seed=seed or seed=0
        img = mi.Bitmap(img)

        # mi.Bitmap(img).write('{:}.ext'.format(scene_name))
        img = img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
        img.write('{:}/{:}_{:}.png'.format(output_folder, scene_name, angle_y))




if __name__ == "__main__":
    main()

