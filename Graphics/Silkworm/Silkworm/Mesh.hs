module Silkworm.Mesh 
  ( Mesh(..)
  , Morph(..)
  , VectorTriple
  , faces
  , blockMorph
  , mkMorph
  , center
  , crSpline
  ) where
  
  import Data.Function (on)
  import Data.List (zip4, groupBy, findIndices)
  import Silkworm.Math (cross, dot, distance3d, average, minus, plus, times, divideby)
  import Numeric.LinearAlgebra hiding (dot)
  
  -- VectorTriple is used for Points, Normals etc.
  type VectorTriple = (Double, Double, Double)
  
  -- A Face is a list of Point/Normal pairs
  type Face = [(VectorTriple, VectorTriple)]
  
  -- A 3D object is a named object and a corresponding list of Faces
  data Mesh = Mesh
                  { objName     :: String              -- Name of the object
                  , objCenter   :: VectorTriple        -- Center of the object
                  , objVertices :: [VectorTriple]      -- Vertices
                  , objNormals  :: [VectorTriple]      -- Normals
                  , objFaceIx   :: [[(Int, Int, Int)]] -- Face indices
                  }
    deriving Show
  
  data Morph = Morph
                  { mphObject   :: Mesh           -- Original unmorphed object
                  , mphCenter   :: VectorTriple       -- Center of the master control points
                  , mphControls :: [VectorTriple]     -- Master control points
                  , mphSlices   :: [[Int]]            -- List of slices; each slice is a list of indices to points
                  }
    | NoMorph
    deriving Show
  
  faces Mesh { objVertices = vs
                 , objNormals  = ns
                 , objFaceIx   = fss
                 } = map (map vnPair) fss
    where vnPair (v, t, n) = (vs !! (v - 1), ns  !! (n - 1))
  
  facePoints fs  = map fst fs
  faceNormals fs = map snd fs
  
  -- | The center of an object
  objectAvgCenter :: Mesh -> (Double, Double, Double)
  objectAvgCenter Mesh { objVertices = vs } = center vs
  
  -- | The center of a list of points
  center :: (Floating a, Floating b, Floating c) => [(a, b, c)] -> (a, b, c)
  center ps = (average xs, average ys, average zs)
    where (xs, ys, zs) = unzip3 ps
  
  -- | mergeAssocs ["one", "two", "three", "four"] [(0, "Duane"), (3, "Kelty")]
  -- | ==> ["Duane","two","three","Kelty"]
  mergeAssocs :: [a] -> [(Int, a)] -> [a]
  mergeAssocs master mergeTable = map merge masterTable
    where
      masterTable  = zip [0..] master
      merge (i, v) = case (lookup i mergeTable) of
                       Just w  -> w
                       Nothing -> v
  
  
  -- blockMorph :: Morph -> [VectorTriple] -> Mesh
  blockMorph m@Morph { mphObject   = obj@Mesh { objVertices = o_vertices
                                                  , objCenter   = o_center
                                                  }
                     , mphCenter   = m_center
                     , mphControls = m_ctrls
                     , mphSlices   = m_slices
                     }
             new_ctrls -- the new set of control points to morph the object around
    |    (length m_slices)
      == (length new_ctrls)    = obj { objVertices = new_vertices }
    where 
      delta = o_center `minus` m_center
      masterTable = zip [0..] o_vertices
      vertexAt i = case (lookup i masterTable) of
                     Just v  -> v
                     Nothing -> error "Lookup failed for masterTable"
      translate v old_ctrl new_ctrl       = v `plus` (new_ctrl `minus` old_ctrl) `minus` delta
      doBlock (slice, old_ctrl, new_ctrl) =
          [(i, translate (vertexAt i) old_ctrl new_ctrl) | i <- slice]
      blocks = map doBlock (zip3 m_slices m_ctrls new_ctrls)
      new_vertices = foldl mergeAssocs o_vertices blocks
  
  -- morph :: Morph -> [VectorTriple] -> Mesh
  -- morph m@(Morph obj slices) cps
  --   | (cpsLen >= 5) && (cpsLen `mod` 2 == 1) = range
  --   | otherwise          = error "Need odd number of control points and at least 5 of them"
  --   where cpsLen         = (length cps)
  --         splineStep     = (fromIntegral ((length cps) - 2)) /
  --                          (fromIntegral (length slices))
  --         splinePoints   = spline splineStep cps
  --         allPoints      = [head cps] ++ splinePoints ++ [last cps]
  --         range          = [1..(length splinePoints)]
  --         (cx, cy, cz)   = splinePoints !! centerPointIdx
  --         centerPointIdx = (length splinePoints) `div` 2
  --         rotatedSlice n = 
  
  -- | Like boundsWithSmudge, but with a default smudge value of 0.0
  bounds = boundsWithSmudge 0.0
  
  -- | Return the minimum and maximum points on a given axis
  boundsWithSmudge smudge axis points = (min - smudge, max + smudge)
    where min = minimum (map axis points)
          max = maximum (map axis points)
  
  mkMorph :: Mesh -> Integer -> Morph
  mkMorph obj@(Mesh { objVertices = o_vertices }) sliceCount =
    Morph { mphObject   = obj
          , mphCenter   = center controls
          , mphControls = controls
          , mphSlices   = slices
          }
    where slices       = (equalParts xAxis sliceCount o_vertices)
          controls     = zip3 axisControls zeros zeros
          axisControls = scanl (+) (min + inc/2) (replicate ((fromInteger sliceCount) - 1) inc)
          zeros        = (cycle [0.0] :: [Double])
          (min, max)   = bounds xAxis o_vertices
          inc          = (max - min) / (fromIntegral sliceCount)
          
  -- lookupPoints vertices ixs = 
  --   where
  -- 
  xAxis (x, y, z) = x
  yAxis (x, y, z) = y
  zAxis (x, y, z) = z
  
  -- | Slices a list of points into n parts and returns a list of lists of indices
  -- | that maps back to the original list of points.
  equalParts :: (Ord a1, Enum a1, Fractional a1, Integral a2) =>
                (a -> a1) -> a2 -> [a] -> [[Int]]
  equalParts axis n points =
      [ findIndices (pointInPartition partPair) points | partPair <- partitions ]
    where
      pointInPartition pair pt = between (axis pt) pair
      between a (a_min, a_max)  = a >= a_min && a < a_max
      partitions = zip (drop 0 range) (drop 1 range)
      range      = [min, (min + inc)..max]
      (min, max) = boundsWithSmudge 0.00001 axis points
      inc        = (max - min) / (fromIntegral n)
      ubound     = (length points) - 1
  
  points2matrix pts = trans m
    where l (x, y, z) = [x, y, z, 1]
          s = foldr1 (++) (map l pts)
          m = ((length pts) >< 4) s
  
  matrix2points m = map tup lists
    where lists = toLists . trans $ m
          tup x = (x !! 0, x !! 1, x !! 2)
  
  rotatePoints i1 i2 f1 f2 pts = map (plus i1) (matrix2points rotated)
    where rotated = rotateVectors (i2 `minus` i1) (f2 `minus` f1) (points2matrix pts)
  
  -- Rotate a matrix of vectors by the difference between two vectors (initial 'i' and final 'f')
  rotateVectors :: VectorTriple -> VectorTriple -> Matrix Double -> Matrix Double
  rotateVectors i f pts = (inv rx)
                       <> (inv ry)
                       <> rz
                       <> ry
                       <> rx
                       <> pts
    where origin    = (0, 0, 0)
          v         = i `cross` f
          v_len     = distance3d origin v
          i_len     = distance3d origin i
          f_len     = distance3d origin f
          (a, b, c) = v `divideby` v_len
          d         = (sqrt $ b**2 + c**2 )
          cs        = (i `dot` f) / (i_len * f_len)
          sn        = v_len / (i_len * f_len)
          rx = (4 >< 4) [    1,    0,     0,    0
                        ,    0,  c/d,  -b/d,    0
                        ,    0,  b/d,   c/d,    0
                        ,    0,    0,     0,    1 ]
          ry = (4 >< 4) [    d,    0,    -a,    0
                        ,    0,    1,     0,    0
                        ,    a,    0,     d,    0
                        ,    0,    0,     0,    1 ]
          rz = (4 >< 4) [   cs,  -sn,     0,    0
                        ,   sn,   cs,     0,    0
                        ,    0,    0,     1,    0
                        ,    0,    0,     0,    1 ]
  
  -- We'll use the catmull-rom spline as a simple default spline
  spline = crSpline
  
  crSplineN :: Integer -> [VectorTriple] -> [VectorTriple]
  crSplineN count pts = take (fromInteger count) $ crSpline step pts
    where step = (fromIntegral ((length pts) - 3)) / (fromIntegral (count - 1))
  
  crSpline :: Double -> [VectorTriple]
              -> [VectorTriple]
  crSpline step pts = map splinePoint time
    where
      splinePoint t = let f = min (floor t) ((length controlPoints) - 1)
                          i = t - (toEnum f)
                      in catmullRom (controlPoints !! f) i
      controlPoints = zip4 (drop 0 pts) (drop 1 pts) (drop 2 pts) (drop 3 pts)
      time          = [0, step .. ubound] :: [Double]
      ubound        = fromIntegral $ length controlPoints
  
  catmullRom :: (Floating t) =>
                ((t, t, t), (t, t, t), (t, t, t), (t, t, t)) -> t -> (t, t, t)
  catmullRom (p0, p1, p2, p3) t = hermite (p1, p2) (m1, m2) t
    where
      m1 = (p2 `minus` p0) `divideby` 2.0
      m2 = (p3 `minus` p1) `divideby` 2.0
  
  kochanekBartel :: (Floating t) =>
                    ((t, t, t), (t, t, t), (t, t, t)) ->
                    t -> t -> t -> t -> (t, t, t)
  kochanekBartel (p0, p1, p2) tens bias cont t = hermite (p1, p2) (m1, m2) t
    where
      m1 = tangent mpp mmm
      m2 = tangent mpm mmp
      mpp = (1 - tens) * (1 + bias) * (1 + cont)
      mmm = (1 - tens) * (1 - bias) * (1 - cont)
      mpm = (1 - tens) * (1 + bias) * (1 - cont)
      mmp = (1 - tens) * (1 - bias) * (1 + cont)
      tangent v1 v2 = ( (v2 `times` (p1 `minus` p0)) `plus` (v1 `times` (p2 `minus` p1)) ) `divideby` 2.0
  
  -- Foundational spline function
  -- (p0, p1) : Two points to connect
  -- (m0, m1) : Corresponding rates of change at those points
  -- t        : An interpolation value within [0,1]
  hermite :: Floating t => ((t, t, t), (t, t, t)) ->
                           ((t, t, t), (t, t, t)) -> t ->
                           (t, t, t)
  hermite (p0, p1) (m0, m1) t = (h00 `times` p0)
                         `plus` (h01 `times` m0)
                         `plus` (h10 `times` p1)
                         `plus` (h11 `times` m1)
    where h00 = (1 + 2 * t) * (1 - t) ** 2
          h01 = (t) * (1 - t) ** 2
          h10 = (t) ** 2 * (3 - 2 * t)
          h11 = (t) ** 2 * (t - 1)
  
  -- Test stuff
  test3Points = ((5.0, 5.0, 0.0)
                ,(5.0, 10.0, 0.0)
                ,(10.0, 10.0, 0.0))
  
  testPoints = [(0.0, 0.0, 0.0)
               ,(5.0, 7.0, 0.0)
               ,(10.0, 0.0, 0.0)
               ,(12.0, 5.0, 0.0)
               ,(15.0, 5.0, 0.0)]
  
  testObject = Mesh {objName = "testObject"
                        ,objCenter = (0, 0, 0)
                        ,objVertices = [(-5.0, -1.0,  0.0)
                                       ,(-5.0,  1.0,  0.0)
                                       ,(-2.5,  1.0,  0.0)
                                       ,(-2.5, -1.0,  0.0)
                                       ,( 0.0, -1.0,  0.0)
                                       ,( 0.0,  1.0,  0.0)
                                       ,( 2.5,  1.0,  0.0)
                                       ,( 2.5, -1.0,  0.0)
                                       ,( 5.0, -1.0,  0.0)
                                       ,( 5.0,  1.0,  0.0)]
                        ,objNormals  = [( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)
                                       ,( 0.0,  0.0,  1.0)]
                        ,objFaceIx   = [[(1,1,1),(2,2,2),(3,3,3),(4,4,4)]
                                       ,[(3,3,3),(4,4,4),(5,5,5),(6,6,6)]
                                       ,[(5,5,5),(6,6,6),(7,7,7),(8,8,8)]
                                       ,[(7,7,7),(8,8,8),(9,9,9),(10,10,10)]]
                        }
                 