module Silkworm.OpenGLHelper where
  
  import Graphics.Rendering.OpenGL
  import Silkworm.Constants (windowDimensions)
  
  -- | Initialize OpenGL to some reasonable values for our game
  initOpenGL :: IO ()
  initOpenGL = do
    -- Use some defaults for drawing with antialiased edges
    clearColor  $= Color4 0.3 0.3 0.3 1.0
    pointSmooth $= Enabled
    pointSize   $= 3
    lineSmooth  $= Enabled
    lineWidth   $= 2.5
    blend       $= Enabled
    blendFunc   $= (SrcAlpha, OneMinusSrcAlpha)
    
    -- Set up the viewport
    matrixMode  $= Projection
    loadIdentity
    ortho x1 x2 y1 y2 (-1) 1
    translate (Vector3 0.5 0.5 0 :: Vector3 GLfloat)
    
    return ()
    
    where
      x1 = -(fromIntegral (fst windowDimensions)) / 2
      x2 =  (fromIntegral (fst windowDimensions)) / 2
      y1 = -(fromIntegral (snd windowDimensions)) / 2
      y2 =  (fromIntegral (snd windowDimensions)) / 2
  