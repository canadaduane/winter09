module Silkworm.OpenGLHelper where
  
  import Graphics.Rendering.OpenGL
  import Silkworm.Constants (windowDimensions)
  
  -- | Initialize OpenGL to some reasonable values for our game
  initOpenGL :: IO ()
  initOpenGL = do
    -- Use some defaults for drawing with antialiased edges
    clearColor  $= Color4 0.3 0.3 0.3 0.0
    -- pointSmooth $= Enabled
    -- pointSize   $= 3
    -- lineSmooth  $= Enabled
    -- lineWidth   $= 2.5
    blend       $= Enabled
    blendFunc   $= (SrcAlpha, OneMinusSrcAlpha)
    shadeModel  $= Smooth
    normalize   $= Enabled
    polygonMode $= (Fill, Fill)
    
    -- Set up the viewport
    viewport    $= (Position 0 0, Size w h)
    matrixMode  $= Projection
    loadIdentity
    ortho (-1) 1 (-1) 1 (-1) 1
    perspective 90.0 (fw/2/(fh/2)) 0.0 20.0
    -- translate (Vector3 0.5 0.5 0 :: Vector3 GLfloat)
    flush
    
    lighting           $= Enabled
    -- position (Light 0) $= Vertex4 0.0 0.0 1.0 0.0
    -- ambient  (Light 0) $= Color4  0.5 0.5 0.5 1.0
    -- light    (Light 0) $= Enabled 
    
    position (Light 1) $= Vertex4 0.0 0.0 (0.5) 0.0
    ambient  (Light 1) $= Color4  0.2 0.2 0.2 1.0
    diffuse  (Light 1) $= Color4  1.0 1.0 1.0 1.0
    specular (Light 1) $= Color4  0.0 0.0 0.0 1.0
    light    (Light 1) $= Enabled
    
    -- materialSpecular  FrontAndBack $= Color4 1.0 1.0 1.0 1.0
    -- materialEmission  FrontAndBack $= Color4 0.0 0.0 0.0 1.0
    -- materialShininess FrontAndBack $= 70.0
    colorMaterial $= Just (Front, Diffuse)
    
    -- matrixMode  $= Projection
    -- loadIdentity
    -- ortho (-1) 1 (-1) 1 (-1) 1
    -- translate (Vector3 0.5 0.5 0 :: Vector3 GLfloat)
    -- 
    -- materialSpecular Front $= Color4 1 1 1 1
    -- materialShininess Front $= 50
    -- position (Light 0) $= Vertex4 1 1 0 1
    -- 
    -- lighting $= Enabled
    -- light (Light 0) $= Enabled
    -- depthFunc $= Just Less
    
    matrixMode $= Modelview 0
    loadIdentity
    flush
    
    return ()
    
    where
      (w, h) = windowDimensions
      (fw, fh) = (fromIntegral w, fromIntegral h) :: (GLdouble, GLdouble)
      x1 = -fw / 2
      x2 =  fw / 2
      y1 = -fh / 2
      y2 =  fh / 2
  