module Silkworm.Object3D where
  
  import Silkworm.Math (distance3d)
  
  -- VectorTriple is used for Points, Normals etc.
  type VectorTriple = (Double, Double, Double)
  
  -- A Face is a list of Point/Normal pairs
  type Face = [(VectorTriple, VectorTriple)]
  
  -- A 3D object is a named object and a corresponding list of Faces
  data Object3D = Object3D String [Face]
    deriving Show
  
  type Influence = [[Double]]
  
  deformation :: Object3D -> [VectorTriple] -> [Influence]
  deformation (Object3D name faces) controls = map fInfluence faces
    where
      invSq d           = 1/d**2
      fInfluence points = map pInfluence points
      pInfluence (p, n) = map (invSq . distance3d p) controls
  
  -- deform :: Object3D -> [Influence] -> [VectorTriple] -> Object3D
  -- deform (Object3D name faces) infs controls = Object3D name newFaces
  --   where
  --     newFaces = 
  
  kochanekBartel :: (Floating t) =>
                    t -> Int -> [(t, t, t)] -> t -> t -> t -> (t, t, t)
  kochanekBartel t i points tension bias continuity =
             ((h0 t) `times` (points !! i))
      `plus` ((h1 t) `times` (startTangent i))
      `plus` ((h2 t) `times` (points !! (i + 1)))
      `plus` ((h3 t) `times` (endTangent i))
    where -- Define some operators on VectorTriples
          times n (x, y, z) = (n*x, n*y, n*z)
          minus (x1, y1, z1) (x2, y2, z2) = (x1-x2, y1-y2, z1-z2)
          plus  (x1, y1, z1) (x2, y2, z2) = (x1+x2, y1+y2, z1+z2)
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
  testObject = Object3D "testObject" [[((-0.1, -0.1, 0.0), (0.0, 0.0, 1.0)),
                                       (( 0.1, -0.1, 0.0), (0.0, 0.0, 1.0)),
                                       (( 0.1,  0.1, 0.0), (0.0, 0.0, 1.0)),
                                       ((-0.1,  0.1, 0.0), (0.0, 0.0, 1.0))]]
  
  