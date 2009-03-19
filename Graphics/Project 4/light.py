from point import Point
from color import Color

class Light:
  def __init__(self, point = Point(), ambient = Color.white, diffuse = Color.white):
    self.enabled = False
    self.point = point
    self.ambient = ambient
    self.diffuse = diffuse
  
  def __str__(self):
    return '[point x:%0.1f, y:%0.1f, z:%0.1f]' % (self.point.x, self.point.y, self.point.z)
  
  def __repr__(self):
    return '[point x:%0.1f, y:%0.1f, z:%0.1f]' % (self.point.x, self.point.y, self.point.z)
