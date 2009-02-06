from numpy import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from copy import copy

from color import Color
from point import Point


# Create a big old matrix with values ranging from 0.0 to 1.0
RASTER_WIDTH = 640
RASTER_HEIGHT = 480
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT * 3
raster = array([ 0.0 for i in range(0, RASTER_SIZE) ], dtype='float32')

clear_color = Color.black
curr_color = Color.white

vertex_mode = 0
vertices = []

point_size = 1
line_width = 1
enabled = {}

def clearColor(r, g, b, a = 1.0):
  """Sets the color that will be used by clear()"""
  global clear_color
  glClearColor(r, g, b, a)
  clear_color = Color(r, g, b, a)

def clear(mode):
  """Fills every pixel on the screen with the color last specified by glClearColor(r,g,b,a)."""
  global raster
  glClear(mode)
  if ((mode & GL_COLOR_BUFFER_BIT) != 0):
    raster = [clear_color.index(i % 3) for i in range(0, RASTER_SIZE)]

def begin(mode):
  global vertices, vertex_mode
  glBegin(mode)
  vertex_mode = mode
  vertices = []

def end():
  global vertices, vertex_mode
  global point_size, line_width, enabled
  glEnd()
  if (vertex_mode == GL_LINE_LOOP):
    color3f(color_first[0], color_first[1], color_first[2])
    vertex2i(vertex_first[0], vertex_first[1], GL_LINE_STRIP)
  vertex_mode = 0

def isEnabled(feature):
  global enabled
  try:
    return enabled[feature]
  except:
    return False

def enable(features):
  global enabled
  glEnable(features)
  enabled[features] = True
  # print enabled

def disable(features):
  global enabled
  glDisable(features)
  enabled[features] = False
  # print enabled

def color3f(r, g, b):
  global curr_color
  glColor3f(r, g, b)
  curr_color = Color(r, g, b)

def color4f(r, g, b, a):
  global curr_color
  glColor4f(r, g, b, a)
  curr_color = Color(r, g, b, a)

def pointSize(s):
  global point_size
  glPointSize(s)
  point_size = s

def lineWidth(w):
  global line_width
  glLineWidth(w)
  line_width = w

def vertex2i(x, y):
  global vertices
  glVertex2i(x, y)
  vertices.append(x, y, 0, curr_color)

def _setPixel(point, color):
  global raster
  pos = ( int(point.y) * RASTER_WIDTH + int(point.x) )*3
  raster[pos+0] = color.r;
  raster[pos+1] = color.g;
  raster[pos+2] = color.b;

def _bresenham_circle(point, color, radius, fn):
  if (radius < 1):
    return
  elif (radius == 1):
    fn(point, color)
  else:
    x = 0
    y = r
    decision = 5.0/4 - r
    while (x <= y):
      fn( Point(point.x + x, point.y + y), color )
      fn( Point(point.x - x, point.y + y), color )
      fn( Point(point.x + x, point.y - y), color )
      fn( Point(point.x - x, point.y - y), color )
      fn( Point(point.x + y, point.y + x), color )
      fn( Point(point.x - y, point.y + x), color )
      fn( Point(point.x + y, point.y - x), color )
      fn( Point(point.x - y, point.y - x), color )
      x = x + 1
      if (decision < 0):
        decision = decision + 2 * x + 1
      else:
        y = y - 1 
        decision = decision + 2 * x + 1 - 2 * y

def _rect(point1, point2, color, fn):
  min_x = min(point1.x, point2.x)
  for x in range(abs(int(point2.x - point1.x))):
    fn( Point(min_x + x, point1.y), color )
    fn( Point(min_x + x, point2.y), color )

  min_y = min(point1.y, point2.y)
  for y in range(abs(int(point2.y - point1.y))):
    fn( Point(point1.x, min_y + y), color )
    fn( Point(point2.x, min_y + y), color )

def _yline(p1, p2, c1, c2, fn = _setPixel):
  # Always draw from bottom to top
  if (p2.y < p1.y):
    p1, p2 = p2, p1
    c1, c2 = c2, c1
  dx = float(p2.x - p1.x)
  dy = float(p2.y - p1.y)
  if (dy != 0):
    gradient = dx/dy
    point = copy(p1)
    color = copy(c1)
    r_inc, g_inc, b_inc = c1.increments(c2, max(abs(dx), abs(dy)))
    while (y < p2.y):
      fn( point, color )
      color.inc(r_inc, g_inc, b_inc)
      point.x += gradient
      point.y += 1

def _hline(p1, p2, c1, c2, fn = _setPixel):
  if (p1.x > p2.x):
    p1, p2 = p2, p1
    c1, c2 = c2, c1
  
  point = copy(p1)
  color = copy(c1)
  r_inc, g_inc, b_inc = c1.increments(c2, x_delta)
  
  for x in range(p1.x, p2.x + 1):
    fn( point, color )
    color.inc(r_inc, g_inc, b_inc)
    point.x += 1
  
def _bresenham_line(p1, p2, c1, c2, fn):
  """Draw's a line from point 1 to point 2, with a color gradient from c1 to c2"""
  # Get our delta values, making sure we stay in quadrant I or IV
  if (p1.x > p2.x):
    p1, p2 = p2, p1
    c1, c2 = c2, c1
  
  y_positive = (p1.y <= p2.y)
  
  x_delta = abs(p2[0] - p1[0])
  y_delta = abs(p2[1] - p1[1])

  point = copy(p1)
  color = copy(c1)
  
  # Generic line-drawing functions that need to be set up properly
  def x_line(xinc, yinc):
    decision = 2 * y_delta - x_delta
    r_inc, g_inc, b_inc = c1.increments(c2, x_delta)
    for i in range(x_delta):
      fn( point, color )
      if (decision < 0):
        decision = decision + 2 * y_delta
      else:
        decision = decision + 2 * (y_delta - x_delta)
        point.y += yinc
      point.x += xinc
      color.inc(r_inc, g_inc, b_inc)
  
  def y_line(xinc, yinc):
    decision = 2 * x_delta - y_delta
    r_inc, g_inc, b_inc = c1.increments(c2, y_delta)
    for i in range(y_delta):
      fn( point, color )
      if (decision < 0):
        decision = decision + 2 * x_delta
      else:
        decision = decision + 2 * (x_delta - y_delta)
        point.x += xinc
      point.y += yinc
      color.inc(r_inc, g_inc, b_inc)
  
  # Determine which "Octant" the line is in
  if (y_positive):
    if (x_delta > y_delta): x_line(1, 1)  # Octant 1
    else:                   y_line(1, 1)  # Octant 2
  else:
    if (x_delta > y_delta): x_line(1, -1) # Octant 8
    else:                   y_line(1, -1) # Octant 7

def _triangle(v1, v2, v3, c1, c2, c3):
  """Draws a shaded triangle from v1 to v2 to v3, with corresponding colors c1, c2, c3"""
  
  bucket = []
  
  def add_point_to_bucket( point, color ):
    bucket.append( [point, color] )
  
  def y_then_x(a, b):
    if (a[0].y > b[0].y):
      return 1;
    elif (a[0].y < b[0].y):
      return -1;
    else:
      if (a[0].x > b[0].x):
        return 1;
      elif (a[0].x < b[0].x):
        return -1;
      else:
        return 0;
  
  if (v1.y != v2.y): # Draw line from v1 to v2
    _yline(v1, v2, c1, c2, add_point_to_bucket)
    
  if (v2.y != v3.y): # Draw line from v2 to v3
    _yline(v2, v3, c2, c3, add_point_to_bucket)
    
  if (v3.y != v1.y): # Draw line from v3 to v1
    _yline(v3, v1, c3, c1, add_point_to_bucket)
  
  if (len(bucket) > 0):
    bucket.sort(y_then_x)
    
    left_point, left_color = bucket.pop(0)
    right_point, right_color = None, None
    for point, color in bucket:
      if (point.y == left_point.y): # same scan line as left_point
        right_point = point
      else:
        if (right_point == None):
          _setPixel( left_point, left_color )
        elif(left_point.x != right_point.x):
          _hline( left_point, right_point, left_color, right_color, _setPixel)
        left_point = point
  
    # Need to complete the final scan line
    _hline( left_point, right_point, left_color, right_color, _setPixel )

