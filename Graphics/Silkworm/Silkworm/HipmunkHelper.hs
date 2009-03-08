module Silkworm.HipmunkHelper where
  
  import Physics.Hipmunk
  import Silkworm.Constants (gravity)
  
  midpoint :: Vector -> Vector -> Vector
  midpoint (Vector x1 y1) (Vector x2 y2) = Vector ((x1 + x2) / 2) ((y1 + y2) / 2)
  
  newStaticLine :: Space -> Vector -> Vector -> IO Body
  newStaticLine space p1 p2 = do
    -- Create a stationary body centered at the midpoint
    body <- newBody infinity infinity
    setPosition body (midpoint p1 p2)
    -- Create the corresponding shape
    shape <- newShape body line 0
    setFriction shape 1.0
    setElasticity shape 0.6
    -- Add it to our physics space
    spaceAdd space (Static shape)
    return body
    where
      line = LineSegment p1 p2 1.0
  
  newSpaceWithGravity :: IO Space
  newSpaceWithGravity = do
    space  <- newSpace
    setGravity space gravity
    return space