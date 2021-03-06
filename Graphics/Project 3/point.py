from numpy import *
from color import Color

class Point:
  def __init__(self, x = 0.0, y = 0.0, z = 0.0, color = Color.white):
    self.x, self.y, self.z = [x, y, z]
    self.color = color
  
  def vector(self):
    return [self.x, self.y, self.z, 1.0]
  
  def xy(self):
    return [self.x, self.y]
  
  def xyz(self):
    return [self.x, self.y, self.z]
  
  def __str__(self):
    return '[x:%.03f, y:%0.3f, z:%.03f, r:%.01f, g:%.01f, b:%.01f]' % (self.x, self.y, self.z, self.color.r, self.color.g, self.color.b)
  
  def __repr__(self):
    return '%s(x:%.03f, y:%.03f, z:%.03f)' % (
      self.__class__.__name__,  # the instance class name
      self.x, self.y, self.z
    )
