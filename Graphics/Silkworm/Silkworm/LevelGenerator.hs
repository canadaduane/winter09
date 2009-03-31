module Silkworm.LevelGenerator (
    rasterizeLines, randomBumpyRaster,
    (#+), (#-), (#*),
    (#+#), (#-#), (#*#)
  ) where
  
  import Data.List (unfoldr)
  import Data.Array.IArray
  import System.Random (StdGen(..), newStdGen, random, split)

  type Point = (Float, Float)
  type Line = (Point, Point)
  type Rect = (Point, Point)
  
  type Point3D = (Float, Float, Float)
  type Plane = (Point3D, Point3D, Point3D)
  
  type Raster = Array (Int, Int) Float
  
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
  
  rasterizeLines :: [Line] -> Rect -> Float -> Raster
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
            [((x, y), -depth (toEnum x, toEnum y)) | x <- [0..x_delta],
                                                    y <- [0..y_delta]]
    where
      distancesToLines :: Point -> [Line] -> [Float]
      distancesToLines p ls = map (distanceToLine p) ls
    
  -- | Combine two rasters by adding elements
  rasterOp :: (Float -> Float -> Float) -> Raster -> Raster -> Raster
  rasterOp op r1 r2 = array (bounds r1) $ map wrapOp $ zip (assocs r1) (assocs r2)
    where wrapOp (((x,y), a), ((_,_), b)) = ((x, y), a `op` b)
  -- rasterOp op r1 r2 = accumArray op 0 (bounds r1) ((assocs r1) ++ (assocs r2))
  
  (#-#) = rasterOp (-)
  a #- b  = amap (- b) a
  (#+#) = rasterOp (+)
  a #+ b  = amap (+ b) a
  (#*#) = rasterOp (*)
  a #* b  = amap (* b) a
  
  randomlist :: Int -> StdGen -> [Int]
  randomlist n = take n . unfoldr (Just . random)
  
  randomBumpyRaster :: Rect -> Int -> Float -> StdGen -> Raster
  randomBumpyRaster rect@((x_min, y_min), (x_max, y_max)) count size gen =
    let x_delta         = ceiling (x_max - x_min)
        y_delta         = ceiling (y_max - y_min)
        lineFromPoint p = (p, p)
        (gen1, gen2)    = split gen
        lines           = map lineFromPoint $
                            zip (map (\x -> fromIntegral $ x `mod` x_delta) (randomlist count gen1))
                                (map (\y -> fromIntegral $ y `mod` y_delta) (randomlist count gen2))
    in rasterizeLines lines rect size
  
  
  
  
  -- Test stuff
  tA = array ((0, 0), (1, 1)) [((0, 0), 0.0), ((0, 1), 5.0), ((1, 0), 10.0), ((1, 1), 20.0)] :: Array (Int, Int) Float
  tB = array ((0, 0), (1, 1)) [((0, 0), 0.0), ((0, 1), 1.0), ((1, 0), 0.0), ((1, 1), 1.0)] :: Array (Int, Int) Float