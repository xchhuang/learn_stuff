#@title imports & utils
import io
import base64
import time
from functools import partial
from typing import NamedTuple
import subprocess

import PIL
import numpy as np
import matplotlib.pylab as pl

# from IPython.display import display, Image, HTML

#@title jax utils
import jax
import jax.numpy as jp


def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

# def imshow(a, fmt='jpeg', display=display):
#   return display(Image(data=imencode(a, fmt)))

# class VideoWriter:
#   def __init__(self, filename='_autoplay.mp4', fps=30.0):
#     self.ffmpeg = None
#     self.filename = filename
#     self.fps = fps
#     self.view = display(display_id=True)
#     self.last_preview_time = 0.0

#   def add(self, img):
#     img = np.asarray(img)
#     h, w = img.shape[:2]
#     if self.ffmpeg is None:
#       self.ffmpeg = self._open(w, h)
#     if img.dtype in [np.float32, np.float64]:
#       img = np.uint8(img.clip(0, 1)*255)
#     if len(img.shape) == 2:
#       img = np.repeat(img[..., None], 3, -1)
#     self.ffmpeg.stdin.write(img.tobytes())
#     t = time.time()
#     if self.view and t-self.last_preview_time > 1:
#        self.last_preview_time = t
#        imshow(img, display=self.view.update)
    
#   def __call__(self, img):
#     return self.add(img)
    
#   def _open(self, w, h):
#     cmd = f'''ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h}
#       -pix_fmt rgb24 -r {self.fps} -i - -pix_fmt yuv420p 
#       -c:v libx264 -crf 20 {self.filename}'''.split()
#     return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

#   def close(self):
#     if self.ffmpeg:
#         self.ffmpeg.stdin.close()
#         self.ffmpeg.wait()
#         self.ffmpeg = None

#   def __enter__(self):
#     return self

#   def __exit__(self, *kw):
#     self.close()
#     if self.filename == '_autoplay.mp4':
#       self.show()

#   def show(self):
#       self.close()
#       if not self.view:
#         return
#       b64 = base64.b64encode(open(self.filename, 'rb').read()).decode('utf8')
#       s = f'''<video controls loop>
#  <source src="data:video/mp4;base64,{b64}" type="video/mp4">
#  Your browser does not support the video tag.</video>'''
#       self.view.update(HTML(s))


# def animate(f, duration_sec, fps=60):
#     with VideoWriter(fps=fps) as vid:
#         for t in jp.linspace(0, 1, int(duration_sec*fps)):
#         vid(f(t))



def norm(v, axis=-1, keepdims=False, eps=0.0):
  return jp.sqrt((v*v).sum(axis, keepdims=keepdims).clip(eps))

def normalize(v, axis=-1, eps=1e-20):
  return v/norm(v, axis, keepdims=True, eps=eps)



def show_slice(sdf, z=0.0, w=400, r=3.5):
    y, x = jp.mgrid[-r:r:w*1j, -r:r:w*1j].reshape(2, -1)
    p = jp.c_[x, y, x*0.0+z]
    d = jax.vmap(sdf)(p).reshape(w, w)
    pl.figure(figsize=(5, 5))
    kw = dict(extent=(-r, r, -r, r), vmin=-r, vmax=r)
    pl.contourf(d, 16, cmap='bwr', **kw );
    pl.contour(d, levels=[0.0], colors='black', **kw);
    pl.axis('equal')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.show()



class Balls(NamedTuple):
  pos: jp.ndarray
  color: jp.ndarray

def create_balls(key, n=16, R=3.0):
  pos, color = jax.random.uniform(key, [2, n, 3])
  pos = (pos-0.5)*R
  return Balls(pos, color)


key = jax.random.PRNGKey(123)
balls = create_balls(key)



def balls_sdf(balls, p, ball_r=0.5):
  dists = norm(p-balls.pos)-ball_r
  return dists.min()

p = jax.random.normal(key, [1000, 3])
# print( jax.vmap(partial(balls_sdf, balls))(p).shape )

# show_slice(partial(balls_sdf, balls), z=0.0);


def scene_sdf(balls, p, ball_r=0.5, c=8.0):
  dists = norm(p-balls.pos)-ball_r
  balls_dist = -jax.nn.logsumexp(-dists*c)/c  # softmin
  floor_dist = p[1]+3.0  # floor is at y==-3.0
  return jp.minimum(balls_dist, floor_dist)  
  
# show_slice(partial(scene_sdf, balls), z=0.0)



def raycast(sdf, p0, dir, step_n=50):
  def f(_, p):
    return p+sdf(p)*dir
  return jax.lax.fori_loop(0, step_n, f, p0)



world_up = jp.array([0., 1., 0.])

def camera_rays(forward, view_size, fx=0.6):
  right = jp.cross(forward, world_up)
  down = jp.cross(right, forward)
  R = normalize(jp.vstack([right, down, forward]))
  w, h = view_size
  fy = fx/w*h
  y, x = jp.mgrid[fy:-fy:h*1j, -fx:fx:w*1j].reshape(2, -1)
  return normalize(jp.c_[x, y, jp.ones_like(x)]) @ R

w, h = 640, 400
pos0 = jp.float32([3.0, 5.0, 4.0])
print('pos0:', pos0)
ray_dir = camera_rays(-pos0, view_size=(w, h))
# sdf = partial(scene_sdf, balls)
# hit_pos = jax.vmap(partial(raycast, sdf, pos0))(ray_dir)
# print(hit_pos.shape)
# print(hit_pos.numpy().min(), hit_pos.numpy().max())
# pl.figure(figsize=(5, 5))
# pl.imshow(np.asarray(hit_pos).reshape(h, w, 3)%1.0)
# pl.show()
# imshow(hit_pos.reshape(h, w, 3)%1.0)
