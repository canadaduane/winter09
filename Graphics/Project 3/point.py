from color import Color

class Point:
  def __init__(self, x = 0.0, y = 0.0, z = 0.0, color = Color(1.0, 1.0, 1.0)):
    self.x, self.y, self.z = [x, y, z]
    self.color = color
  
  def xy(self):
    return [self.x, self.y]
  
  def xyz(self):
    return [self.x, self.y, self.z]
  
  def __str__(self):
    return '[x:%d, y:%d, z:%d]' % (self.x, self.y, self.z)
  
  def __repr__(self):
    return '%s(x:%d, y:%d, z:%d)' % (
      self.__class__.__name__,  # the instance class name
      self.x, self.y, self.z
    )
