from numpy import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from copy import copy

# Create a big old matrix with values ranging from 0.0 to 1.0
RASTER_WIDTH = 640
RASTER_HEIGHT = 480
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT * 3
raster = array([ 0.0 for i in range(0, RASTER_SIZE) ], dtype='float32')

clear_color = [0.0, 0.0, 0.0, 0.0]
color = [1.0, 1.0, 1.0, 1.0]
color_first = [0.0, 0.0, 0.0, 0.0]
color_prev = [1.0, 1.0, 1.0, 1.0]
color_prev2 = [1.0, 1.0, 1.0, 1.0]

vertex_mode = 0
vertex_count = 0
vertex_first = [0, 0]
vertex_prev = [0, 0]
vertex_prev2 = [0, 0]

point_size = 1
line_width = 1
enabled = {}

def clearColor(r, g, b, a = 1.0):
  global clear_color
  glClearColor(r, g, b, a)
  clear_color = [r, g, b, a]

def clear(mode):
  """Fills every pixel on the screen with the color last specified by glClearColor(r,g,b,a)."""
  global raster
  glClear(mode)
  if ((mode & GL_COLOR_BUFFER_BIT) != 0):
    raster = [clear_color[i % 3] for i in range(0, RASTER_SIZE)]

def begin(mode):
  global vertex_mode, vertex_count
  glBegin(mode)
  vertex_count = 0
  vertex_mode = mode

def end():
  global vertex_mode
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
  global color
  glColor3f(r, g, b)
  color = [r, g, b, 1.0]

def color4f(r, g, b, a):
  global color
  glColor4f(r, g, b, a)
  color = [r, g, b, a]

def pointSize(s):
  global point_size
  glPointSize(s)
  point_size = s

def lineWidth(w):
  global line_width
  glLineWidth(w)
  line_width = w

def vertex2i(x, y, mode = False):
  global vertex_count
  global vertex_first, color_first
  global vertex_prev, color_prev
  global vertex_prev2, color_prev2
  global point_size, enabled
  
  if (mode == False):
    mode = vertex_mode
  
  glVertex2i(x, y)
  if (mode == GL_POINTS):
    if (point_size == 1):
      drawPoint2i(x, y)
    else:
      for r in range(int(point_size/2) + 1):
        if (isEnabled(GL_POINT_SMOOTH)):
          drawCircle(x, y, r)
        else:
          drawRect(x, y, r)
  
  elif (mode == GL_LINES):
    if (vertex_count % 2 == 0):
      vertex_prev = [x, y]
      color_prev = copy(color)
    elif (vertex_count % 2 == 1):
      if (line_width == 1):
        drawLine(vertex_prev, [x, y], color_prev, color, drawPoint2iColor)
      else:
        def circle(x, y, r, g, b):
          color3f(r, g, b)
          drawCircle(x, y, int(line_width/2))
        drawLine(vertex_prev, [x, y], color_prev, color, circle)
  
  elif (mode == GL_TRIANGLES):
    pcount = vertex_count % 3
    if (pcount == 0):
      vertex_prev2 = [x, y]
      color_prev2 = copy(color)
    elif (pcount == 1):
      vertex_prev = [x, y]
      color_prev = copy(color)
    elif (pcount == 2):
      drawTriangle(vertex_prev2, vertex_prev, [x, y], color_prev2, color_prev, color)
  
  elif (mode == GL_LINE_STRIP or mode == GL_LINE_LOOP):
    v, c = [x, y], color
    if (vertex_count == 0):
      vertex_first = [x, y]
      color_first = color
    else:
      drawLine(vertex_prev, [x, y], color_prev, color, drawPoint2iColor)
    vertex_prev = v
    color_prev = c
    

  # Always increment the vertex count
  vertex_count += 1

def drawPoint2i(x, y):
  global raster, color
  pos = (int(y) * RASTER_WIDTH + int(x))*3
  raster[pos+0] = color[0];
  raster[pos+1] = color[1];
  raster[pos+2] = color[2];

def drawCircle(x_i, y_i, r):
  if (r < 1):
    return
  elif (r == 1):
    drawPoint2i(x_i, y_i)
  else:
    x = 0
    y = r
    decision = 5.0/4 - r
    while (x <= y):
      drawPoint2i(x_i + x, y_i + y)
      drawPoint2i(x_i - x, y_i + y)
      drawPoint2i(x_i + x, y_i - y)
      drawPoint2i(x_i - x, y_i - y)
      drawPoint2i(x_i + y, y_i + x)
      drawPoint2i(x_i - y, y_i + x)
      drawPoint2i(x_i + y, y_i - x)
      drawPoint2i(x_i - y, y_i - x)
      x = x + 1
      if (decision < 0):
        decision = decision + 2 * x + 1
      else:
        y = y - 1 
        decision = decision + 2 * x + 1 - 2 * y

def drawRect(x_i, y_i, r):
  if (r < 0):
    return
  elif (r == 0):
    drawPoint2i(x_i, y_i)
  else:
    for i in range(r+1):
      drawPoint2i(x_i + r, y_i + i)
      drawPoint2i(x_i - r, y_i + i)
      drawPoint2i(x_i + r, y_i - i)
      drawPoint2i(x_i - r, y_i - i)
      drawPoint2i(x_i + i, y_i + r)
      drawPoint2i(x_i - i, y_i + r)
      drawPoint2i(x_i + i, y_i - r)
      drawPoint2i(x_i - i, y_i - r)

def drawPoint2iColor(x, y, r, g, b):
  color3f(r, g, b)
  drawPoint2i(x, y)

def drawPoint2f(x, y):
  global raster, color
  pos = int(round(y) * RASTER_WIDTH + round(x))*3
  raster[pos+0] = color[0];
  raster[pos+1] = color[1];
  raster[pos+2] = color[2];

def drawPoint2fColor(x, y, r, g, b):
  color3f(r, g, b)
  drawPoint2f(x, y)

def getRGB(c):
  return [float(c[0]), float(c[1]), float(c[2])]

def getRGBinc(c1, c2, dx, dy):
  r1, g1, b1 = getRGB(c1)
  r2, g2, b2 = getRGB(c2)
  r_delta = float(r2) - float(r1)
  g_delta = float(g2) - float(g1)
  b_delta = float(b2) - float(b1)

  def color_inc(length):
    if (length == 0):
      return [0.0, 0.0, 0.0]
    else:
      return [r_delta / length, g_delta / length, b_delta / length]

  return color_inc(max(abs(dx), abs(dy)))
  
def drawPerfectYLine(p1, p2, c1, c2, fn = drawPoint2fColor):
  if (p2[1] < p1[1]):
    p1, p2 = p2, p1
    c1, c2 = c2, c1
  x1, y1 = p1
  x2, y2 = p2
  dx = float(x2 - x1)
  dy = float(y2 - y1)
  if (dy != 0):
    gradient = dx/dy
    r, g, b = getRGB(c1)
    r_inc, g_inc, b_inc = getRGBinc(c1, c2, dx, dy)
    while (y1 < y2):
      fn(x1, y1, r, g, b)
      x1 += gradient
      y1 += 1
      r += r_inc
      g += g_inc
      b += b_inc
  
  
def drawLine(p1, p2, c1, c2, fn):
  """Draw's a line from point 1 to point 2, with a gradient from c1 to c2"""
  
  # print "prev color", c1, "next color", c2
  # Get our delta values, making sure we stay in quadrant I or IV
  if (p1[0] > p2[0]):
    p1, p2 = p2, p1
    c1, c2 = c2, c1
  
  if (p1[1] > p2[1]):
    y_positive = False
  else:
    y_positive = True
  
  r, g, b = getRGB(c1)
  r_inc, g_inc, b_inc = [0.0, 0.0, 0.0]
  r_delta = float(c2[0]) - c1[0]
  g_delta = float(c2[1]) - c1[1]
  b_delta = float(c2[2]) - c1[2]

  x_delta = abs(p2[0] - p1[0])
  y_delta = abs(p2[1] - p1[1])
  
  # Generic line-drawing functions that need to be set up properly
  def x_line(x, y, xinc, yinc, r, g, b):
    decision = 2 * y_delta - x_delta
    for i in range(x_delta):
      fn(x, y, r, g, b)
      if (decision < 0):
        decision = decision + 2 * y_delta
      else:
        decision = decision + 2 * (y_delta - x_delta)
        y += yinc
      x += xinc
      r += r_inc
      g += g_inc
      b += b_inc
  
  def y_line(x, y, xinc, yinc, r, g, b):
    decision = 2 * x_delta - y_delta
    for i in range(y_delta):
      fn(x, y, r, g, b)
      if (decision < 0):
        decision = decision + 2 * x_delta
      else:
        decision = decision + 2 * (x_delta - y_delta)
        x += xinc
      y += yinc
      r += r_inc
      g += g_inc
      b += b_inc
  
  def color_inc(length):
    if (length == 0):
      return [0.0, 0.0, 0.0]
    else:
      return [r_delta / length, g_delta / length, b_delta / length]
  
  # Determine which "Octant" the line is in
  if (y_positive):
    if (x_delta > y_delta): # Octant 1
      r_inc, g_inc, b_inc = color_inc(x_delta)
      x_line(p1[0], p1[1], 1, 1, r, g, b)
    else:                   # Octant 2
      r_inc, g_inc, b_inc = color_inc(y_delta)
      y_line(p1[0], p1[1], 1, 1, r, g, b)
  else:
    if (x_delta > y_delta): # Octant 8
      r_inc, g_inc, b_inc = color_inc(x_delta)
      x_line(p1[0], p1[1], 1, -1, r, g, b)
    else:                   # Octant 7
      r_inc, g_inc, b_inc = color_inc(y_delta)
      y_line(p1[0], p1[1], 1, -1, r, g, b)

def drawTriangle(v1, v2, v3, c1, c2, c3):
  """Draws a shaded triangle from v1 to v2 to v3, with corresponding colors c1, c2, c3"""
  
  bucket = []
  
  def add_point_to_bucket(x, y, r, g, b):
    bucket.append([int(x), int(y), r, g, b])

  def add_point_to_bucket_trace(x, y, r, g, b):
    print "x", x, "y", y
    bucket.append([int(x), int(y), r, g, b])
  
  def y_then_x(a, b):
    if (a[1] > b[1]):
      return 1;
    elif (a[1] < b[1]):
      return -1;
    else:
      if (a[0] > b[0]):
        return 1;
      elif (a[0] < b[0]):
        return -1;
      else:
        return 0;
  
  # v3[0] += 1
  # v3[1] -= 1
  # Draw line from v1 to v2
  if (v1[1] != v2[1]):
    drawPerfectYLine(v1, v2, c1, c2, add_point_to_bucket)
    
  # Draw line from v2 to v3
  if (v2[1] != v3[1]):
    drawPerfectYLine(v2, v3, c2, c3, add_point_to_bucket)
    
  # Draw line from v3 to v1
  if (v3[1] != v1[1]):
    drawPerfectYLine(v3, v1, c3, c1, add_point_to_bucket)
  
  if (len(bucket) > 0):
    bucket.sort(y_then_x)
    
    pt_min = bucket.pop(0)
    pt_max = None
    for pt in bucket:
      if (pt[1] == pt_min[1]): # same scan line as pt_min
        pt_max = pt
      else:
        if (pt_max == None):
          # print "pt_min", pt_min
          apply(drawPoint2iColor, pt_min)
        elif(pt_min[0] != pt_max[0]):
          c1 = [pt_min[2], pt_min[3], pt_min[4], 1.0]
          c2 = [pt_max[2], pt_max[3], pt_max[4], 1.0]
          # print "pt_min", pt_min, "pt_max", pt_max, "c1", c1, "c2", c2
          drawLine(pt_min, pt_max, c1, c2, drawPoint2iColor)
        pt_min = pt
  
    # Need to complete the final scan line
    c1 = [pt_min[2], pt_min[3], pt_min[4], 1.0]
    c2 = [pt_max[2], pt_max[3], pt_max[4], 1.0]
    drawLine(pt_min, pt_max, c1, c2, drawPoint2iColor)
  
  # print bucket