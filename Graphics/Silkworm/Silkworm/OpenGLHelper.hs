module Silkworm.OpenGLHelper
  ( initOpenGL
  , resizeWindow
  , configureProjection
  , lookingAt
  , moveLight
  , PerspectiveType(..)
  ) where
  
  import Graphics.Rendering.OpenGL
  import Graphics.UI.GLFW (windowSize)
  import Silkworm.Constants (windowDimensions)
  import Silkworm.Math (dot, cross, distance3d, minus, plus, times, divideby)
  
  data PerspectiveType = Orthogonal
                       | Perspective Double
  
  -- | Initialize OpenGL to some reasonable values for our game
  initOpenGL :: IO ()
  initOpenGL = do
    -- Use some defaults for drawing with antialiased edges
    clearColor    $= Color4 0.0 0.0 0.0 0.0
    pointSmooth   $= Enabled
    pointSize     $= 3
    lineSmooth    $= Enabled
    lineWidth     $= 2.5
    blend         $= Enabled
    blendFunc     $= (SrcAlpha, OneMinusSrcAlpha)
    depthFunc     $= Just Less
    polygonMode   $= (Fill, Fill)
    texture Texture2D $= Enabled
    
    -- Set up shading and lighting
    lighting      $= Enabled
    shadeModel    $= Smooth -- Flat
    normalize     $= Enabled
    colorMaterial $= Just (Front, Diffuse)
    
    matrixMode $= Modelview 0
    loadIdentity
    normal $ Normal3 0 0 (-1 :: GLfloat)
    return ()
  
  resizeWindow :: Size -> IO ()
  resizeWindow size = do
    configureProjection (Perspective 90.0) (Just size)
    
  
  configureProjection :: PerspectiveType -> Maybe Size -> IO ()
  configureProjection pt explicitSize = do
    matrixMode $= Projection
    loadIdentity
    
    size@(Size w h) <- case explicitSize of
                         Just  s -> return s
                         Nothing -> get windowSize
    viewport      $= (Position 0 0, size)
    let ratio = ((fromIntegral w) / (fromIntegral h))
    
    case pt of
      Orthogonal      -> ortho (-1) 1 (-1) 1 (0.1) 20.0
      Perspective fov -> perspective fov ratio 0.1 20.0
    
    matrixMode $= Modelview 0
    return ()
  
  lookingAt :: Vector3 Double -> Vector3 Double -> Vector3 Double -> IO () -> IO ()
  lookingAt eye@(Vector3 ex ey ez)
            ctr@(Vector3 cx cy cz)
             up@(Vector3 ux uy uz)
            action =
    let f@(fx, fy, fz)     = (cx, cy, cz) `minus` (ex, ey, ez)
        s@(sx, sy, sz)     = fnorm `cross` unorm
        v@(vx, vy, vz)     = s `cross` fnorm
        fnorm              = f `divideby` (distance3d zero f)
        unorm              = (ux, uy, uz) `divideby` (distance3d zero (ux, uy, uz))
        zero               = (0, 0, 0)
    in do
      m <- newMatrix RowMajor [  sx,  sy,  sz,   0
                              ,  vx,  vy,  vz,   0
                              , -fx, -fy, -fz,   0
                              ,   0,   0,   0,   1] :: IO (GLmatrix GLdouble)
      preservingMatrix $ do
        multMatrix m
        translate $ Vector3 (-ex) (-ey) (-ez)
        action
  
  moveLight :: Vector3 Double -> IO ()
  moveLight (Vector3 x y z) = do
    position (Light 1) $= Vertex4 (realToFrac x) (realToFrac y) (realToFrac z) 0.0
    diffuse  (Light 1) $= Color4  1.0 1.0 1.0 1.0
    ambient  (Light 1) $= Color4  0.2 0.2 0.2 1.0
    light    (Light 1) $= Enabled
      