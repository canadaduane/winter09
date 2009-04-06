module Silkworm.Object3D where
  
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
  deform obj (minx, maxx) pts = trace
    where width = maxx - minx
          -- pts = [c1, c2, c3, c4]
          kb t = kochanekBartel t 1 pts 1 1 1
          step = 0.1
          trace = map kb [0,step..1]
  
  spline
  
  splineQuad step pts = map kb [0, step..1]
    where kb t = kochanekBartel t 1 pts 1 1 1
  
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
  
  kochanekBartel :: (Floating t) =>
                    t -> Int -> [(t, t, t)] -> t -> t -> t -> (t, t, t)
  kochanekBartel t i points tension bias continuity =
             ((h0 t) `times` (points !! i))
      `plus` ((h1 t) `times` (startTangent i))
      `plus` ((h2 t) `times` (points !! (i + 1)))
      `plus` ((h3 t) `times` (endTangent i))
    where
          -- Define Hermite basis functions
          h0 t = (1 + 2 * t) * (1 - t) ** 2
          h1 t = (t) * (1 - t) ** 2
          h2 t = (t) ** 2 * (3 - 2 * t)
          h3 t = (t) ** 2 * (t - 1)
          -- Simplify the tension / bias / continuity tangent equations
          mpp = (1 - tension) * (1 + bias) * (1 + continuity)
          mmm = (1 - tension) * (1 - bias) * (1 - continuity)
          mpm = (1 - tension) * (1 + bias) * (1 - continuity)
          mmp = (1 - tension) * (1 - bias) * (1 + continuity)
          tangent i v1 v2 = let p_i  = points !! i
                                p_im = points !! (i - 1)
                                p_ip = points !! (i + 1)
                            in 0.5 `times` ((v1 `times` (p_i  `minus` p_im)) `plus`
                                            (v2 `times` (p_ip `minus` p_i)))
          startTangent i = tangent i mpp mmm
          endTangent   i = tangent i mpm mmp
  
  -- Test stuff
  testPoints = [(0.0, 0.0, 0.0), (5.0, 7.0, 0.0), (10.0, 0.0, 0.0), (15.0, 0.0, 0.0)]
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
  
  