from copy import copy

class Color:
  def __init__(self, r, g, b, a):
    self.r, self.g, self.b, self.a = float(r), float(g), float(b), float(a)
  
  def index(self, i):
    return [self.r, self.g, self.b, self.a][i]
  
  def rgb(self):
    return [self.r, self.g, self.b]
  
  def increments(self, color, steps):
      r_delta = color.r - self.r
      g_delta = color.g - self.g
      b_delta = color.b - self.b
      
      if (steps > 0):
        return [r_delta / length, g_delta / length, b_delta / length]
      else:
        return [0.0, 0.0, 0.0]
  
  def inc(self, r_inc, g_inc, b_inc):
    self.r += r_inc
    self.g += g_inc
    self.b += b_inc
  
Color.white  = Color(1.0, 1.0, 1.0, 1.0)
Color.black  = Color(0.0, 0.0, 0.0, 1.0)
Color.red    = Color(1.0, 0.0, 0.0, 1.0)
Color.green  = Color(0.0, 1.0, 0.0, 1.0)
Color.blue   = Color(0.0, 0.0, 1.0, 1.0)
Color.purple = Color(1.0, 0.0, 1.0, 1.0)
Color.yellow = Color(1.0, 1.0, 0.0, 1.0)
Color.cyan   = Color(0.0, 1.0, 1.0, 1.0)
Color.orange = Color(1.0, 0.5, 0.0, 1.0)
