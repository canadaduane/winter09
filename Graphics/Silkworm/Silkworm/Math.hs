module Silkworm.Math (cross, dot, distance3d) where
  
  -- | Cross product of two 3-dimensional tuples
  cross :: (Floating a) => (a, a, a) -> (a, a, a) -> (a, a, a)
  cross (a1, a2, a3) (b1, b2, b3) = ((a2*b3 - a3*b2), (a3*b1 - a1*b3), (a1*b2 - a2*b1))
  
  dot :: (Floating a) => (a, a, a) -> (a, a, a) -> a
  dot (a1, a2, a3) (b1, b2, b3) = a1*b1 + a2*b2 + a3*b3
  
  distance3d :: (Floating a) => (a, a, a) -> (a, a, a) -> a
  distance3d (ax, ay, az) (bx, by, bz) = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    where dx = bx - ax
          dy = by - ay
          dz = bz - az
    