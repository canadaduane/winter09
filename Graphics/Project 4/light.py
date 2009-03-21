from point import Point
from color import Color

class Light:
  def __init__(self, point = Point(), ambient = Color(1.0, 1.0, 1.0), diffuse = Color(1.0, 1.0, 1.0)):
    self.enabled = False
    self.point = point
    self.ambient = ambient
    self.diffuse = diffuse
  
  def __str__(self):
    return '[point x:%0.1f, y:%0.1f, z:%0.1f : r:%0.1f, g:%0.1f, b:%0.1f]' % \
      (self.point.x, self.point.y, self.point.z,\
       self.diffuse.r, self.diffuse.g, self.diffuse.b)
  
  def __repr__(self):
    return '[point x:%0.1f, y:%0.1f, z:%0.1f : r:%0.1f, g:%0.1f, b:%0.1f]' % \
      (self.point.x, self.point.y, self.point.z,\
       self.diffuse.r, self.diffuse.g, self.diffuse.b)
