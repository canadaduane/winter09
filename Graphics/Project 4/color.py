from numpy import array
from copy import copy

class Color:
  def __init__(self, r, g, b, a = 1.0):
    self.r, self.g, self.b, self.a = float(r), float(g), float(b), float(a)
  
  def index(self, i):
    return [self.r, self.g, self.b, self.a][i]
  
  def rgb(self):
    return [self.r, self.g, self.b]
  
  def vector(self):
    return array([self.r, self.g, self.b, 1.0])
  
  def set(self, r, g, b, a = 1.0):
    self.r, self.g, self.b, self.a = r, g, b, a
  
  def increments(self, color, steps):
      r_delta = color.r - self.r
      g_delta = color.g - self.g
      b_delta = color.b - self.b
      
      if (abs(steps) > 0):
        return [r_delta / steps, g_delta / steps, b_delta / steps]
      else:
        return [0.0, 0.0, 0.0]
  
  def inc(self, r_inc, g_inc, b_inc):
    self.r += r_inc
    self.g += g_inc
    self.b += b_inc
  
  def __str__(self):
    return '[r:%0.3f, g:%0.3f, b:%0.3f]' % (self.r, self.g, self.b)
  
  def __repr__(self):
    return '%s(r:%0.3f, g:%0.3f, b:%0.3f)' % (
      self.__class__.__name__,  # the instance class name
      self.r, self.g, self.b
    )
  
Color.white  = Color(1.0, 1.0, 1.0, 1.0)
Color.black  = Color(0.0, 0.0, 0.0, 1.0)
Color.red    = Color(1.0, 0.0, 0.0, 1.0)
Color.green  = Color(0.0, 1.0, 0.0, 1.0)
Color.blue   = Color(0.0, 0.0, 1.0, 1.0)
Color.purple = Color(1.0, 0.0, 1.0, 1.0)
Color.yellow = Color(1.0, 1.0, 0.0, 1.0)
Color.cyan   = Color(0.0, 1.0, 1.0, 1.0)
Color.orange = Color(1.0, 0.5, 0.0, 1.0)
