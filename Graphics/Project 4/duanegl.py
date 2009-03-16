from numpy import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from copy import copy, deepcopy

from color import Color
from point import Point

class Viewport:
  def __init__(self, xmin, ymin, width, height):
    self.xmin, self.ymin = float(xmin), float(ymin)
    self.width, self.height = float(width), float(height)

def n_at_a_time(list_, n):
    return [list_[i:i+n] for i in xrange(0, len(list_), n)]

# Create a big old matrix with values ranging from 0.0 to 1.0
RASTER_WIDTH = 640
RASTER_HEIGHT = 480
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT * 3
raster = array([ 0.0 for i in range(0, RASTER_SIZE) ], dtype='float32')
depth_buffer = array([ 10000 for i in range(0, RASTER_WIDTH * RASTER_HEIGHT) ], dtype='float32')

clear_color = Color.black
curr_color = Color.white
curr_normal = Normal.default

vport = Viewport(0, 0, 640, 480)
identity_matrix = matrix(array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='float32'))
vertex_mode = 0
matrix_mode = GL_PROJECTION
vertices = []
projection_matrix_stack = [copy(identity_matrix)]
modelview_matrix_stack = [copy(identity_matrix)]

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
  global raster, depth_buffer
  glClear(mode)
  if ((mode & GL_COLOR_BUFFER_BIT) != 0):
    raster = [clear_color.index(i % 3) for i in range(0, RASTER_SIZE)]
  if ((mode & GL_DEPTH_BUFFER_BIT) != 0):
    depth_buffer = array([ 10000 for i in range(0, RASTER_WIDTH * RASTER_HEIGHT) ], dtype='float32')

def matrixMode(mode):
  global matrix_mode, projection_matrix_stack, modelview_matrix_stack
  
  glMatrixMode(mode)
  matrix_mode = mode

def loadIdentity():
  global matrix_mode, projection_matrix_stack, modelview_matrix_stack
  glLoadIdentity()
  if matrix_mode == GL_PROJECTION:
    projection_matrix_stack[-1] = copy(identity_matrix)
  elif matrix_mode == GL_MODELVIEW:
    modelview_matrix_stack[-1] = copy(identity_matrix)

def loadMatrix(v16):
  global matrix_mode, projection_matrix_stack, modelview_matrix_stack
  glLoadMatrixd(v16)
  if matrix_mode == GL_PROJECTION:
    projection_matrix_stack[-1] = matrix(v16).reshape(4, 4)
  elif matrix_mode == GL_MODELVIEW:
    modelview_matrix_stack[-1] = matrix(v16).reshape(4, 4)

def multMatrix(v16, useGl = True):
  global matrix_mode, projection_matrix_stack, modelview_matrix_stack
  if useGl:
    glMultMatrixd(v16)
  if matrix_mode == GL_PROJECTION:
    projection_matrix_stack[-1] = projection_matrix_stack[-1] * matrix(copy(v16)).reshape(4, 4)
  elif matrix_mode == GL_MODELVIEW:
    modelview_matrix_stack[-1] = modelview_matrix_stack[-1] * matrix(copy(v16)).reshape(4, 4)

def pushMatrix():
  global matrix_mode, projection_matrix_stack, modelview_matrix_stack
  glPushMatrix()
  if matrix_mode == GL_PROJECTION:
    newmatrix = copy(projection_matrix_stack[-1])
    projection_matrix_stack.append(newmatrix)
  elif matrix_mode == GL_MODELVIEW:
    newmatrix = copy(modelview_matrix_stack[-1])
    modelview_matrix_stack.append(newmatrix)

def popMatrix():
  global matrix_mode, projection_matrix_stack, modelview_matrix_stack
  glPopMatrix();
  if matrix_mode == GL_PROJECTION:
    projection_matrix_stack.pop(-1)
  elif matrix_mode == GL_MODELVIEW:
    modelview_matrix_stack.pop(-1)
  
def begin(mode):
  global vertices, vertex_mode
  glBegin(mode)
  vertex_mode = mode
  vertices = []

def end():
  global vertices, vertex_mode
  glEnd()

  # Call the appropriate "end" function
  {
    GL_POINTS:         _end_points,
    GL_LINES:          _end_lines,
    GL_TRIANGLES:      _end_triangles,
    GL_LINE_STRIP:     _end_line_strip,
    GL_LINE_LOOP:      _end_line_loop,
    GL_TRIANGLE_STRIP: _end_triangle_strip,
    GL_TRIANGLE_FAN:   _end_triangle_fan,
    GL_QUADS:          _end_quads,
    GL_QUAD_STRIP:     _end_quad_strip,
    GL_POLYGON:        _end_triangle_fan
  }[vertex_mode](transformed(vertices))
  
  vertices = []
  vertex_mode = 0

def transformed(vertices):
  p = projection_matrix_stack[-1]
  m = modelview_matrix_stack[-1]
  vvecs = matrix([v.vector() for v in vertices]).transpose()
  intermediate = (p * m * vvecs).transpose()
  # print "p", p
  # print "m", m
  # print "vvecs", vvecs
  # print "intermediate", intermediate
  
  # Divide by w
  for i in range(len(intermediate)):
    intermediate[i] = intermediate[i] / intermediate[i,3]
  
  vw = vport.width/2
  vh = vport.height/2
  def viewport_transform(vector, point):
    x = vector[0,0] * vw + vw + vport.xmin
    y = vector[1,0] * vh + vh + vport.ymin
    z = vector[2,0]
    return Point(x, y, z, point.color, point.normal)
  
  values = [ viewport_transform(intermediate[i].transpose(), vertices[i]) \
             for i in range(len(vertices)) ]
  return values

def _set_pixel(point):
  global raster, depth_buffer
  x = int(point.x)
  y = int(point.y)
  z = point.z
  if x >= vport.xmin and x < vport.width and \
     y >= vport.ymin and y < vport.height:
    one_d = ( y * RASTER_WIDTH + x )
    pos = one_d * 3
    if z < depth_buffer[one_d] and z >= -1.0 and z <= 1.0:
      depth_buffer[one_d] = z
      raster[pos+0] = point.color.r;
      raster[pos+1] = point.color.g;
      raster[pos+2] = point.color.b;

def _end_points(vertices):
  global point_size
  
  for v in vertices:
    _point( v, point_size, isEnabled(GL_POINT_SMOOTH) )

def _end_lines(vertices):
  global line_width
  
  def draw_point( p ):
    _point( p, line_width, True )
  
  for p1, p2 in n_at_a_time( vertices, 2 ):
    _bresenham_line( p1, p2, draw_point )

def _end_triangles(vertices):
  for p1, p2, p3 in n_at_a_time( vertices, 3 ):
    _triangle( p1, p2, p3, _set_pixel )

def _end_line_strip(vertices):
  global line_width
  
  def draw_point( p ):
    _point( p, line_width, True )
  
  p1 = vertices[0]
  for p2 in vertices[1:len(vertices)]:
    _bresenham_line( p1, p2, draw_point )
    p1 = p2

def _end_line_loop(vertices):
  global line_width

  def draw_point( p ):
    _point( p, line_width, True )
  
  _end_line_strip()
  p1 = vertices[0]
  p2 = vertices[len(vertices)-1]
  _bresenham_line( p1, p2, draw_point )

def _end_triangle_strip(vertices):
  n = 0
  for p1, p2, p3 in n_at_a_time( vertices, 3 ):
    if (n % 2 == 0):
      _triangle( p1, p2, p3, _set_pixel )
    else:
      _triangle( p2, p3, p1, _set_pixel )
    n += 1

def _end_triangle_fan(vertices):
  p1, p2 = vertices[0:2]
  for p3 in vertices[2:len(vertices)]:
    _triangle( p1, p2, p3, _set_pixel )
    p2 = p3

def _end_quads(vertices):
  for p1, p2, p3, p4 in n_at_a_time( vertices, 4 ):
    _triangle( p1, p2, p3, _set_pixel )
    _triangle( p3, p1, p4, _set_pixel )
  

def _end_quad_strip(vertices):
  vertices.reverse()
  p1, p2 = vertices[0:2]
  for p3, p4 in n_at_a_time(vertices[2:len(vertices)], 2):
    _triangle( p1, p2, p4, _set_pixel )
    _triangle( p3, p4, p1, _set_pixel )
    p1 = p3
    p2 = p4


def isEnabled(feature):
  global enabled
  try:    return enabled[feature]
  except: return False

def enable(features):
  global enabled
  glEnable(features)
  enabled[features] = True

def disable(features):
  global enabled
  glDisable(features)
  enabled[features] = False

def normal3f(x, y, z):
  global curr_normal
  glNormal3f(x, y, z)
  curr_normal = Normal(x, y, z)

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
  vertices.append( Point(x, y, 0, curr_color, curr_normal) )

def vertex2f(x, y):
  global vertices
  glVertex2f(x, y)
  vertices.append( Point(x, y, 0, curr_color, curr_normal) )

def vertex3f(x, y, z):
  global vertices
  glVertex3f(x, y, z)
  vertices.append( Point(x, y, z, curr_color, curr_normal) )

def _point( point, size = 1, smooth = False ):
  if (size <= 0):
    pass
  elif (size == 1):
    _set_pixel( point )
  else:
    half = size / 2
    if (smooth):
      _circle_filled( point, half, _set_pixel )
    else:
      ul = Point( point.x - half, point.y + half, 0.0, point.color, point.normal )
      lr = Point( point.x + half, point.y - half, 0.0, point.color, point.normal )
      _rect_filled( ul, lr, _set_pixel )

def _circle(point, radius, fn):
  if (radius < 1):
    return
  elif (radius == 1):
    fn(point, color)
  else:
    x = 0
    y = radius
    decision = 5.0/4 - radius
    while (x <= y):
      fn( Point(point.x + x, point.y + y, 0.0, color, point.normal) )
      fn( Point(point.x - x, point.y + y, 0.0, color, point.normal) )
      fn( Point(point.x + x, point.y - y, 0.0, color, point.normal) )
      fn( Point(point.x - x, point.y - y, 0.0, color, point.normal) )
      fn( Point(point.x + y, point.y + x, 0.0, color, point.normal) )
      fn( Point(point.x - y, point.y + x, 0.0, color, point.normal) )
      fn( Point(point.x + y, point.y - x, 0.0, color, point.normal) )
      fn( Point(point.x - y, point.y - x, 0.0, color, point.normal) )
      x = x + 1
      if (decision < 0):
        decision = decision + 2 * x + 1
      else:
        y = y - 1 
        decision = decision + 2 * x + 1 - 2 * y

def _circle_filled(point, radius, fn):
  bucket = []
  
  def add_point_to_bucket( point ):
    bucket.append( deepcopy( point ) )
  
  _circle(point, radius, add_point_to_bucket)
  
  _fill( bucket, fn )


def _rect(point1, point2, fn):
  min_x = min(point1.x, point2.x)
  for x in range(abs(int(point2.x - point1.x))):
    fn( Point(min_x + x, point1.y, 0.0, color, point.normal) )
    fn( Point(min_x + x, point2.y, 0.0, color, point.normal) )

  min_y = min(point1.y, point2.y)
  for y in range(abs(int(point2.y - point1.y))):
    fn( Point(point1.x, min_y + y, 0.0, color, point.normal) )
    fn( Point(point2.x, min_y + y, 0.0, color, point.normal) )

def _rect_filled(point1, point2, fn):
  bucket = []
  
  def add_point_to_bucket( point ):
    bucket.append( deepcopy( point ) )
  
  _rect(point1, point2, add_point_to_bucket)
  
  _fill( bucket, fn )

def _yline(p1, p2, fn = _set_pixel):
  # Always draw from bottom to top
  if (p2.y < p1.y):
    p1, p2 = p2, p1
  dx = float(p2.x - p1.x)
  dy = float(p2.y - p1.y)
  dz = float(p2.z - p1.z)
  if (dy != 0):
    gradient = dx/dy
    point = deepcopy(p1)
    r_inc, g_inc, b_inc = p1.color.increments(p2.color, abs(dy))
    z_inc = dz / dy
    while (point.y < p2.y):
      fn( point )
      point.color.inc(r_inc, g_inc, b_inc)
      point.x += gradient
      point.y += 1
      point.z += z_inc

def _hline(p1, p2, fn = _set_pixel):
  if (p1.x > p2.x):
    p1, p2 = p2, p1
  
  dx = p2.x - p1.x
  dz = float(p2.z - p1.z)
  point = deepcopy(p1)
  r_inc, g_inc, b_inc = p1.color.increments(p2.color, dx)
  z_inc = dz / dx
  for x in range(int(p1.x), int(p2.x) + 1):
    fn( point )
    point.color.inc(r_inc, g_inc, b_inc)
    point.x += 1
    point.z += z_inc
  
def _bresenham_line(p1, p2, fn):
  """Draw's a line from point 1 to point 2, with a color gradient from c1 to c2"""
  # Get our delta values, making sure we stay in quadrant I or IV
  if (p1.x > p2.x):
    p1, p2 = p2, p1
  
  y_positive = (p1.y <= p2.y)
  
  x_delta = int(abs(p2.x - p1.x))
  y_delta = int(abs(p2.y - p1.y))
  z_delta = float(abs(p2.z - p1.z))
  
  point = deepcopy(p1)
  
  # Generic line-drawing functions that need to be set up properly
  def x_line(xinc, yinc, zinc):
    decision = 2 * y_delta - x_delta
    r_inc, g_inc, b_inc = p1.color.increments(p2.color, x_delta)
    for i in range(x_delta):
      fn( point )
      if (decision < 0):
        decision = decision + 2 * y_delta
      else:
        decision = decision + 2 * (y_delta - x_delta)
        point.y += yinc
      point.x += xinc
      point.z += zinc
      point.color.inc(r_inc, g_inc, b_inc)
  
  def y_line(xinc, yinc, zinc):
    decision = 2 * x_delta - y_delta
    r_inc, g_inc, b_inc = p1.color.increments(p2.color, y_delta)
    for i in range(y_delta):
      fn( point )
      if (decision < 0):
        decision = decision + 2 * x_delta
      else:
        decision = decision + 2 * (x_delta - y_delta)
        point.x += xinc
      point.z += zinc
      point.y += yinc
      point.color.inc(r_inc, g_inc, b_inc)
  
  z_inc = z_delta / sqrt(x_delta ** 2 + y_delta ** 2)
  # Determine which "Octant" the line is in
  if (y_positive):
    if (x_delta > y_delta): x_line(1, 1, z_inc)  # Octant 1
    else:                   y_line(1, 1, z_inc)  # Octant 2
  else:
    if (x_delta > y_delta): x_line(1, -1, z_inc) # Octant 8
    else:                   y_line(1, -1, z_inc) # Octant 7

def _triangle(v1, v2, v3, fn = _set_pixel):
  """Draws a shaded triangle from v1 to v2 to v3"""
  
  bucket = []
  
  def add_point_to_bucket( point ):
    bucket.append( deepcopy( point ) )
  
  if (v1.y != v2.y): # Draw line from v1 to v2
    _yline(v1, v2, add_point_to_bucket)
  else:
    _hline(v1, v2, add_point_to_bucket)
  
  if (v2.y != v3.y): # Draw line from v2 to v3
    _yline(v2, v3, add_point_to_bucket)
  else:
    _hline(v2, v3, add_point_to_bucket)
  
  if (v3.y != v1.y): # Draw line from v3 to v1
    _yline(v3, v1, add_point_to_bucket)
  else:
    _hline(v3, v1, add_point_to_bucket)
  
  _fill( bucket, fn )

def _fill(points, fn):
  def y_then_x(a, b):
    if (a.y > b.y):     return 1
    elif (a.y < b.y):   return -1
    else:
      if (a.x > b.x):   return 1
      elif (a.x < b.x): return -1
      else:             return 0
  
  if (len(points) > 0):
    points.sort(y_then_x)

    left_point = points.pop(0)
    right_point = None
    for point in points:
      if (int(point.y) == int(left_point.y)): # same scan line as left_point
        right_point = point
      else:
        if (right_point == None):
          _set_pixel( left_point )
        elif(int(left_point.x) != int(right_point.x)):
          _hline( left_point, right_point, fn)
        left_point = point

    # Need to complete the final scan line
    _hline( left_point, right_point, fn )
  
def viewport(xmin, ymin, width, height):
  global vport
  glViewport(xmin, ymin, width, height)
  vport = Viewport(xmin, ymin, width, height)

def rotate(deg, x, y, z):
  glRotatef(deg, x, y, z)
  theta = float(deg)/180*pi
  m = sqrt(x**2 + y**2 + z**2)
  a = float(x) / m
  b = float(y) / m
  c = float(z) / m
  d = sqrt(b**2 + c**2)
  rx = matrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, c/d, -b/d, 0.0],
    [0.0, b/d, c/d, 0.0],
    [0.0, 0.0, 0.0, 1.0]])
  ry = matrix([
    [d, 0.0, -a, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [a, 0.0, d, 0.0],
    [0.0, 0.0, 0.0, 1.0]])
  rz = matrix([
    [cos(theta), -sin(theta), 0.0, 0.0],
    [sin(theta), cos(theta), 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])
  multMatrix(rx.I, False)
  multMatrix(ry.I, False)
  multMatrix(rz, False)
  multMatrix(ry, False)
  multMatrix(rx, False)

def translate(tx, ty, tz):
  multMatrix(matrix([
    [1.0, 0.0, 0.0, float(tx)],
    [0.0, 1.0, 0.0, float(ty)],
    [0.0, 0.0, 1.0, float(tz)],
    [0.0, 0.0, 0.0, 1.0]]))

def scale(sx, sy, sz):
  multMatrix(matrix([
    [float(sx), 0.0, 0.0, 0.0],
    [0.0, float(sy), 0.0, 0.0],
    [0.0, 0.0, float(sz), 0.0],
    [0.0, 0.0, 0.0, 1.0]]))

def ortho(l, r, b, t, n, f):
  multMatrix(matrix([
    [2.0/float(r-l), 0.0, 0.0, float(r+l)/float(r-l)],
    [0.0, 2.0/float(t-b), 0.0, float(t+b)/float(t-b)],
    [0.0, 0.0, -2.0/float(f-n), float(f+n)/float(f-n)],
    [0.0, 0.0, 0.0, 1.0]]).transpose())

def shear(sxy, sxz, syx, syz, szx, szy):
  multMatrix(matrix([
    [1.0, float(sxy), float(sxz), 0.0],
    [float(syx), 1.0, float(syz), 0.0],
    [float(szx), float(szy), 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]))

def fixed_scale(sx, sy, sz,  cx, cy, cz):
  translate(cx, cy, cz)
  multMatrix(matrix([
    [float(sx), 0.0, 0.0, 0.0],
    [0.0, float(sy), 0.0, 0.0],
    [0.0, 0.0, float(sz), 0.0],
    [0.0, 0.0, 0.0, 1.0]]))
  translate(-cx, -cy, -cz)

def frustum(left, right, bottom, top, znear, zfar):
  a = (right+left) / (right-left)
  b = (top+bottom) / (top-bottom)
  c = (zfar-znear) / (zfar-znear)
  d = (2*zfar*znear) / (zfar-znear)
  
  multMatrix(matrix([
    [(2*znear)/(right-left), 0, a, 0],
    [0, (2*znear)/(top-bottom), b, 0],
    [0, 0, c, d],
    [0, 0, -1, 0]]))

def perspective(fovy, aspect, znear, zfar):
   ymax = znear * tan(fovy * math.pi / 360.0)
   ymin = -ymax
   xmin = ymin * aspect
   xmax = ymax * aspect
   
   frustum(xmin, xmax, ymin, ymax, znear, zfar)

