module Silkworm.LevelGenerator (rasterizeLines) where
  import Data.Array.IArray
  
  type Point = (Float, Float)
  type Line = (Point, Point)
  type Rect = (Point, Point)
  
  type Point3D = (Float, Float, Float)
  type Plane = (Point3D, Point3D, Point3D)
  
  data Failure = Failure
  
  -- Weights the distance between two 2D points by a multiple of x or y
  weightedDistance :: Float -> Float -> Point -> Point -> Float
  weightedDistance xWeight yWeight (x1, y1) (x2, y2) = sqrt (xWeight * sqDx + yWeight * sqDy)
    where dx    = x2 - x1
          sqDx  = dx ** 2
          dy    = y2 - y1
          sqDy  = dy ** 2
  
  xWeightedDistance :: Float -> Point -> Point -> Float
  xWeightedDistance xWeight p1 p2 = weightedDistance xWeight 1.0 p1 p2
  
  yWeightedDistance :: Float -> Point -> Point -> Float
  yWeightedDistance yWeight p1 p2 = weightedDistance 1.0 yWeight p1 p2

  -- Default x-weighted distance (x weight = 2.0)
  xwDistance :: Point -> Point -> Float
  xwDistance p1 p2 = xWeightedDistance 2.0 p1 p2
  
  -- Default y-weighted distance (y weight = 2.0)
  ywDistance :: Point -> Point -> Float
  ywDistance p1 p2 = yWeightedDistance 2.0 p1 p2
  
  -- Standard pythagorean distance between two 2D points
  distance :: Point -> Point -> Float
  distance p1 p2 = weightedDistance 1.0 1.0 p1 p2
  
  -- http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/
  distanceToLine :: Point -> Line -> Float
  distanceToLine p0@(x0, y0) (p1@(x1, y1), p2@(x2, y2)) = 
    let line_m      = distance p1 p2
        pt_dx       = x0 - x1
        pt_dy       = y0 - y1
        line_dx     = x2 - x1
        line_dy     = y2 - y1
        u           = ( pt_dx * line_dx + pt_dy * line_dy ) / ( line_m ** 2 )
        p_intersect = (x1 + u * line_dx, y1 + u * line_dy) in
      if u >= 0.0 && u <= 1.0
        -- An intersection point along our line exists, so calculate distance to it
        then distance p0 p_intersect
        -- A line perpendicular to our line that crosses through p0 does not exist,
        -- so use the endpoints of our line to calculate distance instead.
        else min (distance p0 p1) (distance p0 p2)
  
  -- | Decompose a list of lines into a list of points
  linesToPoints :: [Line] -> [Point]
  linesToPoints lines = (map fst lines) ++ (map snd lines)
  
  -- | Given a list of lines, return two points that describe a bounding box around those lines
  bbox :: [Line] -> Rect
  bbox lines = ((x_min, y_min), (x_max, y_max))
    where
      points       = linesToPoints lines
      xs           = (map fst points)
      ys           = (map snd points)
      x_min        = minimum xs
      x_max        = maximum xs
      y_min        = minimum ys
      y_max        = maximum xs
  
  -- | Square root of the sum of all numbers squared
  rootSum :: (Floating a) => [a] -> a
  rootSum ns = sqrt $ foldl (+) 0 $ map (** 2) ns
  
  rasterizeLines :: [Line] -> Rect -> Float -> Array (Int, Int) Float
  rasterizeLines lines ((x_min, y_min), (x_max, y_max)) size =
    let x_delta      = ceiling (x_max - x_min)
        y_delta      = ceiling (y_max - y_min)
        depth (x, y) =
          let dists     = distancesToLines (x + x_min, y + y_min) lines
              dist_min  = minimum dists
              toRad d   = d / size * (pi / 2)
              -- r_sum     = toRad $ rootSum dists
              r_min     = toRad dist_min
              -- r_best    = max (cos r_min) (cos r_sum)
          in  if dist_min > size
                then 0
                else cos r_min * size
    in
      -- Create an array of size (x_delta x y_delta), filling it with 
      array ((0, 0), (x_delta, y_delta))
            [((x, y), depth (toEnum x, toEnum y)) | x <- [0..x_delta],
                                                    y <- [0..y_delta]]
    where
      distancesToLines :: Point -> [Line] -> [Float]
      distancesToLines p ls = map (distanceToLine p) ls
    
  
  -- generateTunnels :: [Point] -> [Line]
  -- generateTunnels pts = []
  --   where
  -- Sorts two pairs of points using a function such as xwDistance or ywDistance
  -- lowest :: (Point -> Point -> Float) -> (Point, Point) -> (Point, Point) -> Ordering
  -- lowest distFn (p1, p2) (p3, p4) = compare (distFn p1 p2) (distFn p3 p4)
  -- 
  -- nextClosest :: (Point -> Point -> Float) -> Point -> [Point] -> (Point, [Point])
  -- nextClosest distFn pt pts =
  --   let ptPairs = zip (repeat pt) pts
  --       (first:others) = sortBy (lowest distFn) ptPairs in
  --   (snd first, map snd others)
  