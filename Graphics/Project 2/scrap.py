def drawPerfectLine(p1, p2, c1, c2):
  x1, y1 = p1
  x2, y2 = p2
  r1, g1, b1, _ = getRGBf(c1)
  r2, g2, b2, _ = getRGBf(c2)
  r, g, b = r1, g1, b1
  r_delta = float(r2) - float(r1)
  g_delta = float(g2) - float(g1)
  b_delta = float(b2) - float(b1)
  dx = x2 - x1
  dy = y2 - y1
  gradient = dx/dy
  ey = int(y1+1) - y1
  ex = gradient*ey
  ax = x1 + ex
  ay = int(y1+1)
  by = int(y2)
  
  def color_inc(length):
    if (length == 0):
      return [0.0, 0.0, 0.0]
    else:
      return [r_delta / length, g_delta / length, b_delta / length]
  
  if (abs(dx) > abs(dy)):
    r_inc, g_inc, b_inc = color_inc(dx)
  else:
    r_inc, g_inc, b_inc = color_inc(dy)
  
  x = ax
  y = ay
  ybuffer = []
  if (by > ay):
    while (y < by):
      drawPointColor(int(x), int(y), r, g, b)
      ybuffer.append(x)
      x += gradient
      y += 1
      r += r_inc
      g += g_inc
      b += b_inc
  else:
    while (y > by):
      drawPointColor(int(x), int(y), r, g, b)
      ybuffer.append(x)
      x += gradient
      y -= 1
      r += r_inc
      g += g_inc
      b += b_inc
  
  print ybuffer






  def _rgb_inc(c1, c2, dx, dy):
    r_delta = c2.r - c1.r
    g_delta = c2.g - c1.g
    b_delta = c2.b - c1.b

    def inc(length):
      if (length == 0): return [0.0, 0.0, 0.0]
      else:             return [r_delta / length, g_delta / length, b_delta / length]

    return inc(max(abs(dx), abs(dy)))

    if (mode == GL_POINTS):
      if (point_size == 1):
        _setPixel(x, y)
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
          drawLine(vertex_prev, [x, y], color_prev, color, _setPixelColor)
        else:
          def circle(x, y, r, g, b):
            color3f(r, g, b)
            drawCircle(x, y, int(line_width/2))
          drawLine(vertex_prev, [x, y], color_prev, color, circle)

    elif (mode == GL_TRIANGLES or mode == GL_TRIANGLE_STRIP):
      pcount = vertex_count % 3
      if (pcount == 0):
        vertex_prev2 = [x, y]
        color_prev2 = copy(color)
      elif (pcount == 1):
        vertex_prev = [x, y]
        color_prev = copy(color)
      elif (pcount == 2):
        if (mode == GL_TRIANGLE_STRIP):
          n = vertex_count / 3
          if (n % 2 == 1):
            drawTriangle(vertex_prev2, vertex_prev, [x, y], color_prev2, color_prev, color)
          else:
            drawTriangle(vertex_prev, [x, y], vertex_prev2, color_prev, color, color_prev2)
        else:
          drawTriangle(vertex_prev2, vertex_prev, [x, y], color_prev2, color_prev, color)

    elif (mode == GL_LINE_STRIP or mode == GL_LINE_LOOP):
      v, c = [x, y], color
      if (vertex_count == 0):
        vertex_first = [x, y]
        color_first = color
      else:
        drawLine(vertex_prev, [x, y], color_prev, color, _setPixelColor)
      vertex_prev = v
      color_prev = c