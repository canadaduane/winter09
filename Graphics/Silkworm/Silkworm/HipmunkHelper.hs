module Silkworm.HipmunkHelper where
  
  import Silkworm.Math
  import Physics.Hipmunk
  
  instance Show Shape where
    show s = "Shape"
  
  instance Show Space where
    show s = "Space"
  
  midpoint :: Vector -> Vector -> Vector
  midpoint (Vector x1 y1) (Vector x2 y2) = Vector ((x1 + x2) / 2) ((y1 + y2) / 2)
  
  -- | The center of a list of points
  vecCenter :: [Vector] -> Vector
  vecCenter ps = Vector (average xs) (average ys)
    where xs = map (\(Vector x _) -> x) ps
          ys = map (\(Vector _ y) -> y) ps
  
  
  area :: ShapeType -> Float
  area (Circle r) = pi * (r ** 2)
  area (LineSegment p1 p2 thick) = 0
  area (Polygon vs) = w * h
    where w = (maximum $ map getx vs) - (minimum $ map getx vs)
          h = (maximum $ map gety vs) - (minimum $ map gety vs)
          getx (Vector x y) = x
          gety (Vector x y) = y
  
  newSpaceWithGravity :: Gravity -> IO Space
  newSpaceWithGravity g = do
    space  <- newSpace
    setElasticIterations space 10
    setGravity space g
    return space