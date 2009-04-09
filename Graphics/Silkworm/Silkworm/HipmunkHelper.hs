module Silkworm.HipmunkHelper where
  
  import Physics.Hipmunk
  import Silkworm.Constants (gravity, elasticity)
  
  midpoint :: Vector -> Vector -> Vector
  midpoint (Vector x1 y1) (Vector x2 y2) = Vector ((x1 + x2) / 2) ((y1 + y2) / 2)
  
  newSpaceWithGravity :: IO Space
  newSpaceWithGravity = do
    space  <- newSpace
    setElasticIterations space 10
    setGravity space gravity
    return space