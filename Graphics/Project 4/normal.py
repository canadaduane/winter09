from copy import copy

class Color:
  def __init__(self, x, y, z):
    self.x, self.y, self.z = [x, y, z]
  
  def __str__(self):
    return '[x:%0.1f, y:%0.1f, z:%0.1f]' % (self.x, self.y, self.z)
  
  def __repr__(self):
    return '%s(x:%0.1f, y:%0.1f, z:%0.1f)' % (
      self.__class__.__name__,  # the instance class name
      self.x, self.y, self.z
    )

Normal.default = Normal(1.0, 1.0, 1.0)
