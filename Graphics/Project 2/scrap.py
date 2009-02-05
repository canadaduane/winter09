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