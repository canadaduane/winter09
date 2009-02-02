from numpy import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Create a big old matrix with values ranging from 0.0 to 1.0
RASTER_WIDTH = 640
RASTER_HEIGHT = 480
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT * 3
raster = array([float(i)/RASTER_SIZE for i in range(0, RASTER_SIZE)], dtype='float32')

clear_color = [0.0, 0.0, 0.0, 0.0]
color = [1.0, 1.0, 1.0, 1.0]
vertex_mode = 0
vertex_count = 0
vertex_prev = [0, 0]

def clearColor(r, g, b, a = 1.0):
  global clear_color
  clear_color = [r, g, b, a]

def clear(mode):
  """Fills every pixel on the screen with the color last specified by glClearColor(r,g,b,a)."""
  global raster
  if ((mode & GL_COLOR_BUFFER_BIT) != 0):
    raster = [clear_color[i % 3] for i in range(0, RASTER_SIZE)]

def color3f(r, g, b):
  global color
  color = [r, g, b, 1.0]

def vertex2i(x, y):
  global vertex_count
  if (vertex_mode == GL_POINTS):
    drawPoint(x, y)
  elif (vertex_mode == GL_LINES):
    if (vertex_count % 2 == 0):
      vertex_prev = [x, y]
    elif (vertex_count % 2 == 1):
      drawLine(vertex_prev, [x, y])
  elif (vertex_mode == GL_TRIANGLES):
    raise RuntimeError, "GL_TRIANGLES unimplemented"
    
  # Always increment the vertex count
  vertex_count += 1

def begin(mode):
  global vertex_mode, vertex_count
  if (mode != GL_POINTS and mode != GL_LINES and mode != GL_TRIANGLES):
    raise RuntimeError, "DuaneGL.begin accepts only GL_POINTS, GL_LINES and GL_TRIANGLES"
  vertex_count = 0
  vertex_mode = mode

def end():
  global vertex_mode
  vertex_mode = 0

def drawPoint(x, y):
  global raster, color
  pos = (y * RASTER_WIDTH + x)*3
  raster[pos+0] = color[0];
  raster[pos+1] = color[1];
  raster[pos+2] = color[2];
