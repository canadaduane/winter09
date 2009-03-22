from numpy import *
from color import Color
from normal import Normal

class Point:
  def __init__(self, x = 0.0, y = 0.0, z = 0.0, color = Color.white, n = Normal.default):
    self.x, self.y, self.z = [x, y, z]
    self.color = color
    self.normal = n
  
  def vector(self):
    return array([self.x, self.y, self.z, 1.0])
  
  def xy(self):
    return [self.x, self.y]
  
  def xyz(self):
    return [self.x, self.y, self.z]
  
  def set(self, x, y, z, n = 1.0):
    self.x, self.y, self.z = x, y, z
  
  def lit(self, lp):
    def unit(vec):
      return vec / linalg.norm(vec)
    delta = lp.vector() - self.vector()
    n = self.normal.vector()
    print "delta", delta, "n", n
    return max(0, dot(n, unit(delta)))
  
  def __str__(self):
    return '[x:%.03f, y:%0.3f, z:%.03f, r:%.01f, g:%.01f, b:%.01f]' % (self.x, self.y, self.z, self.color.r, self.color.g, self.color.b)
  
  def __repr__(self):
    return '%s(x:%.03f, y:%.03f, z:%.03f)' % (
      self.__class__.__name__,  # the instance class name
      self.x, self.y, self.z
    )
