module Silkworm.Object3D where
  
  import Data.Function (on)
  import Data.List (zip4, groupBy, findIndices)
  import Silkworm.Math (cross, dot, distance3d)
  import Numeric.LinearAlgebra hiding (dot)
  
  -- VectorTriple is used for Points, Normals etc.
  type VectorTriple = (Double, Double, Double)
  
  -- A Face is a list of Point/Normal pairs
  type Face = [(VectorTriple, VectorTriple)]
  
  -- A 3D object is a named object and a corresponding list of Faces
  data Object3D = Object3D String [VectorTriple]   -- points
                                  [VectorTriple]   -- normals
                                  [[(Int, Int, Int)]] -- face indices
    deriving Show
  
  type Influence = [[Double]]
  
  faces (Object3D name vs ns fss) = map (map vnPair) fss
    where vnPair (v, t, n) = (vs !! (v - 1), ns  !! (n - 1))
  
  facePoints fs  = map fst fs
  faceNormals fs = map snd fs
  
  -- Define some operators on VectorTriples
  times n (x, y, z) = (n*x, n*y, n*z)
  divideby vt n = times (1/n) vt
  minus (x1, y1, z1) (x2, y2, z2) = (x1-x2, y1-y2, z1-z2)
  plus  (x1, y1, z1) (x2, y2, z2) = (x1+x2, y1+y2, z1+z2)
  
  -- deform :: Object3D -> (Double, Double) ->
  --           VectorTriple -> VectorTriple -> VectorTriple -> VectorTriple ->
  --           Object3D
  -- deform obj (minx, maxx) pts = trace
  --   where width = maxx - minx
  --         -- pts = [c1, c2, c3, c4]
  --         kb t = kochanekBartel t 1 pts 1 1 1
  --         step = 0.1
  --         trace = map kb [0,step..1]
  
  -- splineQuad step pts = map kb [0, step..1]
  --   where kb t = kochanekBartel t 1 pts 1 1 1
  
  -- deformation :: Object3D -> [VectorTriple] -> [Influence]
  -- deformation (Object3D name faces) controls = map fInfluence faces
  --   where
  --     invSq d           = 1/d**2
  --     fInfluence points = map pInfluence points
  --     pInfluence (p, n) = map (invSq . distance3d p) controls
  
  -- deform :: Object3D -> [Influence] -> [VectorTriple] -> Object3D
  -- deform (Object3D name faces) infs controls = Object3D name newFaces
  --   where
  --     newFaces = 
  
  xAxis (x, y, z) = x
  yAxis (x, y, z) = y
  zAxis (x, y, z) = z
  equalParts :: (Ord a1, Enum a1, Fractional a1, Integral a2) =>
                (a -> a1) -> a2 -> [a] -> [[Int]]
  equalParts axisFn parts points =
      [ findIndices (pointInPartition p) points | p <- partitions ]
    where
      pointInPartition pair pt = between (axisFn pt) pair
      between a (m, n)  = a >= m && a <= n
      partitions = zip (drop 0 range) (drop 1 range)
      range      = [minima, (minima+inc)..maxima]
      minima     = minimum (map axisFn points)
      maxima     = maximum (map axisFn points)
      ubound     = (length points) - 1
      inc        = (maxima - minima) / (fromIntegral parts)
  
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
  
  crSpline :: (Enum t, Floating t) => t -> [(t, t, t)] -> [(t, t, t)]
  crSpline step pts = concatMap (flip map $ time) controlFns
    where
      time = [0, step..1]
      controlPoints = zip4 (drop 0 pts) (drop 1 pts) (drop 2 pts) (drop 3 pts)
      controlFns = map catmullRom controlPoints
  
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
  test3Points = ((5.0, 5.0, 0.0), (5.0, 10.0, 0.0), (10.0, 10.0, 0.0))
  testPoints = [(25.0, 0.0, 0.0), (0.0, 0.0, 0.0), (5.0, 7.0, 0.0), (10.0, 0.0, 0.0),
                (12.0, 5.0, 0.0), (15.0, 5.0, 0.0)]
  testControls = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.0)]
  testObject = Object3D "testObject" [(-0.1, -0.1,  0.0)
                                     ,( 0.1, -0.1,  0.0)
                                     ,( 0.1,  0.1,  0.0)
                                     ,(-0.1,  0.1,  0.0)]
                                     [( 0.0,  0.0,  1.0)
                                     ,( 0.0,  0.0,  1.0)
                                     ,( 0.0,  0.0,  1.0)
                                     ,( 0.0,  0.0,  1.0)]
                                     [[(1,1,1),(2,2,2),(3,3,3),(4,4,4)]]
  
  