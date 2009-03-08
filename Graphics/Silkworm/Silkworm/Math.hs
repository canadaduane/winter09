module Silkworm.Math (cross) where
  
  -- | Cross product of two 3-dimensional tuples
  cross :: (Float, Float, Float) -> (Float, Float, Float) -> (Float, Float, Float)
  cross (a1, a2, a3) (b1, b2, b3) = ((a2*b3 - a3*b2), (a3*b1 - a1*b3), (a1*b2 - a2*b1))
  