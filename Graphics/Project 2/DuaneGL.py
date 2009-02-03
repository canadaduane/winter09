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
  global vertex_count, vertex_prev
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

def drawLine(p1, p2):
  """Draw's a line from point 1 to point 2, using the current color"""
  global raster, color
  
  # Get our delta values, making sure we stay in quadrant I or IV
  if (p1[0] > p2[0]):
    p1, p2 = p2, p1
  
  if (p1[1] > p2[1]):
    y_positive = False
  else:
    y_positive = True
  
  x_delta = abs(p2[0] - p1[0])
  y_delta = abs(p2[1] - p1[1])
  
  # Generic line-drawing functions that need to be set up properly
  def x_line(x, y, xinc, yinc):
    decision = 2 * y_delta - x_delta
    for i in range(x_delta - 1):
      drawPoint(x, y)
      if (decision < 0):
        decision = decision + 2 * y_delta
      else:
        decision = decision + 2 * (y_delta - x_delta)
        y += yinc
      x += xinc
  
  def y_line(x, y, xinc, yinc):
    decision = 2 * x_delta - y_delta
    for i in range(y_delta - 1):
      # print "x", x, "y", y
      drawPoint(x, y)
      if (decision < 0):
        decision = decision + 2 * x_delta
      else:
        decision = decision + 2 * (x_delta - y_delta)
        x += xinc
      y += yinc
  
  # Determine which "Octant" the line is in
  if (y_positive):
    if (x_delta > y_delta): # Octant 1
      print "oc1 x", p1[0], "y", p1[1], "x_delta", x_delta, "y_delta", y_delta
      x_line(p1[0], p1[1], 1, 1)
    else:                   # Octant 2
      print "oc2 x", p1[0], "y", p1[1], "x_delta", x_delta, "y_delta", y_delta
      y_line(p1[0], p1[1], 1, 1)
  else:
    if (x_delta > y_delta): # Octant 8
      print "oc8 x", p1[0], "y", p1[1], "x_delta", x_delta, "y_delta", y_delta
      x_line(p1[0], p1[1], 1, -1)
    else:                   # Octant 7
      print "oc7 x", p1[0], "y", p1[1], "x_delta", x_delta, "y_delta", y_delta
      y_line(p1[0], p1[1], 1, -1)
