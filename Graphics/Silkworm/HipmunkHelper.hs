module HipmunkHelper where
  
  import Phyiscs.Hipmunk
  
  midpoint :: Vector -> Vector -> Vector
  midpoint (Vector x1 y1) (Vector x2 y2) = Vector ((x1 + x2) / 2) ((y1 + y2) / 2)
  
  newStaticLine :: Vector -> Vector -> Vector -> IO Body
  newStaticLine p1 p2 =
    static <- newBody infinity infinity
    setPosition static (midpoint p1 p2)