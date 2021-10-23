import taichi as ti
import numpy as np
import math
import matplotlib.pyplot as plt

real = ti.f32
# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu)

# sz = 800
spp = 100
res_wh = 400
res = (res_wh, res_wh)
aspect_ratio = res[0] / res[1]

eps = 1e-4
inf = 1e10
fov = 0.8
max_ray_depth = 10

mat_none = 0
mat_lambertian = 1
mat_light = 2

camera_pos = ti.Vector([0.0, 0.6, 3.0])

# light_color = ti.Vector([0.9, 0.85, 0.7])
light_color = ti.Vector([1, 1, 1])
color_buffer = ti.Vector(3, dt=ti.f32, shape=res)
random_samples = ti.field(dtype=real, shape=(res_wh, res_wh, spp, 3), needs_grad=True)
camera_samples = ti.field(dtype=real, shape=(res_wh, res_wh, spp, 2), needs_grad=True)
L = ti.field(dtype=real, shape=(), needs_grad=True)

random_samples.from_numpy(np.random.rand(res_wh, res_wh, spp, 3))
camera_samples.from_numpy(np.random.rand(res_wh, res_wh, spp, 2))


# print(random_samples.shape, camera_samples.shape)


@ti.func
def ray_sphere_intersect(ray_o, ray_d, center, radius):
    t = inf
    oc = ray_o - center
    a = ray_d.norm_sqr()
    b = 2 * ti.dot(oc, ray_d)
    c = oc.norm_sqr() - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        t = inf
    else:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
    return t


@ti.func
def ray_plane_intersect(ray_o, ray_d, pt_on_plane, norm):
    t = inf
    denom = ti.dot(ray_d, norm)
    if abs(denom) > eps:
        t = ti.dot((pt_on_plane - ray_o), norm) / denom
    return t


@ti.func
def intersect_scene(ray_o, ray_d):
    closest, normal = inf, ti.Vector.zero(ti.f32, 3)
    color, mat = ti.Vector.zero(ti.f32, 3), mat_none

    ## left
    pnorm = ti.Vector([1.0, 0.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([-1.1, 0.0, 0.0]), pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = ti.Vector([0.65, 0.05, 0.05]), mat_lambertian
    ## right
    pnorm = ti.Vector([-1.0, 0.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([1.1, 0.0, 0.0]), pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = ti.Vector([0.12, 0.45, 0.15]), mat_lambertian
    # bottom
    # gray = ti.Vector([0.93, 0.93, 0.93])
    gray = ti.Vector([0.8, 0.8, 0.8])
    pnorm = ti.Vector([0.0, 1.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([0.0, 0.0, 0.0]), pnorm)

    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = ti.Vector([0.2, 0.2, 0.2]), mat_lambertian
    # top
    pnorm = ti.Vector([0.0, -1.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([0.0, 2.0, 0.0]), pnorm)
    # cur_dist = ray_sphere_intersect(ray_o, ray_d, ti.Vector([0.0, 1.5, 0.2]), 0.2)

    # print(cur_dist)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = gray, mat_light
    # far
    pnorm = ti.Vector([0.0, 0.0, 1.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([0.0, 0.0, 0.0]), pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = ti.Vector([0.2, 0.2, 0.2]), mat_lambertian

    # obstacle
    cur_dist = ray_sphere_intersect(ray_o, ray_d, ti.Vector([0.0, 1.2, 1.5]), 0.2)

    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = gray, mat_lambertian

    return closest, normal, color, mat


@ti.func
def random_in_unit_sphere():
    p = ti.Vector([0.0, 0.0, 0.0])
    while True:
        for i in ti.static(range(3)):
            p[i] = ti.random() * 2 - 1.0
        if p.dot(p) <= 1.0:
            break
    return p


@ti.func
def random_in_unit_hemisphere():
    # p = ti.Vector([0.0, 0.0, 0.0])
    sample_x = ti.random()
    sample_y = ti.random()
    z = ti.random()

    r = ti.sqrt(1 - z * z)
    phi = 2 * np.pi * sample_y
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    return ti.Vector([x, y, z])
    # return p


def random_in_unit_sphere_np():
    p = np.array([0, 0, 0])
    while True:
        for i in range(3):
            p[i] = np.random.rand() * 2 - 1.0
        if p.dot(p) <= 1.0:
            break
    return p


@ti.func
def sample_ray_dir(u, v, cur_spp, indir, normal, hit_pos, mat):
    # p = random_in_unit_sphere()
    p = random_in_unit_hemisphere()
    p = ti.Vector([0.0, 0.0, 0.0])
    p[0] = random_samples[u, v, cur_spp, 0]
    p[1] = random_samples[u, v, cur_spp, 1]
    p[2] = random_samples[u, v, cur_spp, 2]

    # print(p)
    return ti.normalized(p + normal)


@ti.kernel
def generate_random_samples_in_unit_hemisphere():
    # for _ in range(1):
    for u, v in ti.ndrange((0, res_wh), (0, res_wh)):
        for cur_spp in range(spp):
            p = random_in_unit_hemisphere()
            # print('ind:', ind, random_samples.shape)
            # random_samples[ind+0] = 0.0
            # print(ind, random_samples.shape)
            random_samples[u, v, spp, 0] = p.x
            random_samples[u, v, spp, 1] = p.y
            random_samples[u, v, spp, 2] = p.z


@ti.kernel
def render(cur_spp: ti.i32):
    for u, v in color_buffer:
        ray_pos = camera_pos
        ray_dir = ti.Vector([
            (2 * fov * (u + camera_samples[u, v, cur_spp, 0]) / res[1] - fov * aspect_ratio),
            (2 * fov * (v + camera_samples[u, v, cur_spp, 0]) / res[1] - fov),
            -1.0,
        ])

        ray_dir = ti.normalized(ray_dir)

        px_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        # first hit the scene
        closest, hit_normal, hit_color, mat = intersect_scene(ray_pos, ray_dir)
        # # if hit the emitter
        if mat == mat_light:
            px_color = throughput * light_color
        #
        #
        # # else continue shooting next ray
        # for depth in range(10):
        hit_pos = ray_pos + closest * ray_dir
        # print('hit_pos:', hit_pos)
        ray_dir = sample_ray_dir(u, v, cur_spp, ray_dir, hit_normal, hit_pos, mat)
        # ray_dir = ti.normalized(ti.Vector([0.0, 2.0, 0.0]) - hit_pos)
        # print(ray_dir)
        ray_pos = hit_pos + eps * ray_dir
        throughput *= hit_color
        # print('hit_color:', hit_color)
        #
        closest, hit_normal, hit_color, mat = intersect_scene(ray_pos, ray_dir)
        if mat == mat_light:
            px_color = throughput * light_color
        color_buffer[u, v] += px_color

        # depth = 0
        # while depth < max_ray_depth:
        #     # for depth in range(1):
        #     closest, hit_normal, hit_color, mat = intersect_scene(ray_pos, ray_dir)
        #     if mat == mat_none:
        #         break
        #     if mat == mat_light:
        #         px_color = throughput * light_color
        #         break
        #     hit_pos = ray_pos + closest * ray_dir
        #     depth += 1
        #     ray_dir = sample_ray_dir(ray_dir, hit_normal, hit_pos, mat)
        #     ray_pos = hit_pos + eps * ray_dir
        #     throughput *= hit_color
        #
        # color_buffer[u, v] += px_color


generate_random_samples_in_unit_hemisphere()
# print(random_samples.shape)
# generate_random_samples_in_unit_sphere()


gui = ti.GUI('Cornell Box', res)
for i in range(spp):
    # with ti.Tape(loss=L):
    render(i)
    # print(random_samples.grad[0, 0, i, 0])
    img = color_buffer.to_numpy(as_vector=True) * (1 / (i + 1))
    img = np.sqrt(img / img.mean() * 0.24)
    # img = np.sqrt(img / img.mean() * 1)

    # print(img.shape)
    # plt.figure(1)
    # plt.imshow(img)
    # plt.show()
    gui.set_image(img)
    gui.show()
#
input('input')
