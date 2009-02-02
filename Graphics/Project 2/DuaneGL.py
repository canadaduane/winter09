from numpy import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Create a big old matrix with values ranging from 0.0 to 1.0
RASTER_WIDTH = 640
RASTER_HEIGHT = 480
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT * 3
raster = array([float(i)/RASTER_SIZE for i in range(0, RASTER_SIZE)], dtype='float32')

dj_clear_color = [0.0, 0.0, 0.0, 0.0]
dj_color = [1.0, 1.0, 1.0, 1.0]
dj_vertex_mode = 0

def djClearColor(r, g, b, a = 1.0):
  global dj_clear_color
  dj_clear_color = [r, g, b, a]

def djClear(mode):
  """Fills every pixel on the screen with the color last specified by glClearColor(r,g,b,a)."""
  global raster
  if ((mode & GL_COLOR_BUFFER_BIT) != 0):
    raster = [dj_clear_color[i % 3] for i in range(0, RASTER_SIZE)]

def djColor3f(r, g, b):
  global dj_color
  dj_color = [r, g, b, 1.0]

def djVertex2i(x, y):
  if (dj_vertex_mode == GL_POINTS):
    djDrawPoint(x, y)
  elif (dj_vertex_mode == GL_LINES):
    raise RuntimeError, "unimplemented"
  elif (dj_vertex_mode == GL_TRIANGLES):
    raise RuntimeError, "unimplemented"
  
def djBegin(mode):
  global dj_vertex_mode
  if (mode != GL_POINTS and mode != GL_LINES and mode != GL_TRIANGLES):
    raise RuntimeError, "djBegin accepts only GL_POINTS, GL_LINES and GL_TRIANGLES"
  dj_vertex_mode = mode

def djEnd():
  global dj_vertex_mode
  dj_vertex_mode = 0

def djDrawPoint(x, y):
  global raster, dj_color
  pos = (y * RASTER_WIDTH + x)*3
  raster[pos+0] = dj_color[0];
  raster[pos+1] = dj_color[1];
  raster[pos+2] = dj_color[2];
