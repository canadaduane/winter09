from color import Color

class Point:
  def __init__(self, x = 0.0, y = 0.0, z = 0.0, color = Color.white):
    self.x, self.y, self.z = [x, y, z]
    self.color = color
  
  def xy(self):
    return [self.x, self.y]
  
  def xyz(self):
    return [self.x, self.y, self.z]
