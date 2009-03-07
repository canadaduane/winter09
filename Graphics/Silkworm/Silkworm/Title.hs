module Silkworm.Title where
  
  import System.Random (newStdGen, split)
  import System.Directory (getCurrentDirectory)
  import System.Exit (ExitCode(..), exitWith)
  import Control.Monad (forM, forM_)
  import qualified Data.Map as Map
  import Data.IORef (IORef, newIORef)
  import Data.Array.IArray as Array
  import Graphics.UI.GLFW (
    Key(..), SpecialKey(..),
    BitmapFont(..), renderString,
    swapBuffers)
  import Graphics.Rendering.OpenGL (
    GLint, GLfloat,
    ClearBuffer(..), clear,
    Vector3(..), scale, translate, rotate,
    Vertex3(..), vertex,
    Color3(..), color,
    Normal3(..), normal,
    PrimitiveMode(..), renderPrimitive,
    MatrixMode(..), matrixMode, loadIdentity,
    preservingMatrix,
    ($=))
  import Graphics.Rendering.OpenGL.GL.StateVar (get)
  import Graphics.Rendering.FTGL (
    RenderMode(..), Font, createTextureFont, createPolygonFont,
    setFontFaceSize, renderFont, getFontError)
  import Silkworm.WindowHelper (getPressedKeys, keyIsPressed)
  import Silkworm.ImageHelper (loadTexture, renderTexture)
  import Silkworm.LevelGenerator (
    rasterizeLines, randomBumpyRaster,
    (#+), (#-), (#*),
    (#+#), (#-#), (#*#))
  
  type KeyMap = Map.Map Key Bool
  
  data TitleObject = TitleObject {
    tPosition   :: (GLfloat, GLfloat),
    tPoints     :: [(GLfloat, GLfloat)],
    tRadians    :: Float,
    tTextureId  :: Integer
  }
  
  data TitleState = TitleState {
    tsObjects :: [TitleObject],
    tsAngle   :: Float,
    tsFont    :: Font
  }
  
  rasterRect = ((0, 0), (200, 200))
  testTunnel = rasterizeLines [((50,50), (100,130)), ((50,100), (150,100))] rasterRect 20.0
  
  cross :: (Float, Float, Float) -> (Float, Float, Float) -> (Float, Float, Float)
  -- cross (x0, y0, z0) (x1, y1, z1) = ((y0*z1 - z0*y1), (z0*x1 - x0*z1), (x0*y1 - x1*y0))
  cross (a1, a2, a3) (b1, b2, b3) = ((a2*b3 - a3*b2), (a3*b1 - a1*b3), (a1*b2 - a2*b1))
  
  drawTunnel :: Array.Array (Int, Int) Float -> Float -> IO ()
  drawTunnel t angle = preservingMatrix $ do
    rotate angle $ Vector3 1 0.5 (0 :: GLfloat)
    scale 2.0 2.0 (2.0 :: Float)
    translate (Vector3 (-0.5) (-0.5) (0.0 :: GLfloat))
    color $ Color3 0.5 0.5 (1.0 :: GLfloat) 
    let ((x1, y1), (x2, y2)) = bounds t
        rng = range ((x1, y1), (x2 - 1, y2 - 1))
        shrink n = (fromIntegral n) / 200
        -- v x y z = vertex (Vertex3 (shrink x) (shrink y) z / 200.0)
        xyz x y = (shrink x, shrink y, (t ! (x, y)) / 200.0)
      in
      forM_ rng $ \(x, y) -> renderPrimitive Quads $ do
        let (x0, y0, z0) = xyz x y
            (x1, y1, z1) = xyz (x + 1) y
            (x2, y2, z2) = xyz x (y + 1)
            (x3, y3, z3) = xyz (x + 1) (y + 1)
            (nx, ny, nz) = cross (x1 - x0, y1 - y0, z1 - z0) (x2 - x0, y2 - y0, z2 - z0)
        normal $ Normal3 nx ny nz
        vertex $ Vertex3 x0 y0 z0
        vertex $ Vertex3 x1 y1 z1
        vertex $ Vertex3 x3 y3 z3
        vertex $ Vertex3 x2 y2 z2
        
        -- vertex $ Vertex3 1.0 1.0 (1.0 :: Float)
        -- let xf = fromIntegral x
        --     yf = fromIntegral y
        --   in vertex $ Vertex3 (xf :: GLfloat) yf (t ! (x, y))
        --      vertex $ Vertex3 (xf + 1) yf (t ! (x + 1, y))
        --      vertex $ Vertex3 (xf + 1) (yf + 1) (t ! (x + 1, y + 1))
        --      vertex $ Vertex3 xf (yf + 1) (t ! (x, y + 1))
  
  -- | Return a TitleState object set to reasonable initial values
  newTitleState :: IO TitleState
  newTitleState = do
    -- bgImage <- readImage "background.png"
    -- let objs = [TitleObject (0, 0) [] 0.0 bgImage]
    let objs = []
    font <- createPolygonFont "scribbled.ttf"
    -- putStrLn (show $ getFontError font)
    -- exitWith (ExitFailure (-1))
    return (TitleState objs 0.0 font)
  
  showTitleScreen :: IO ()
  showTitleScreen = do
    loadResources
    state <- newTitleState >>= newIORef
    titleScreenLoop state
  
  titleScreenLoop :: IORef TitleState -> IO ()
  titleScreenLoop stateVar = do
    keySet <- getPressedKeys [SpecialKey ESC]
    updateDisplay stateVar
    if keyIsPressed (SpecialKey ESC) keySet
      then return ()
      else titleScreenLoop stateVar
  
  -- | Load title screen textures into OpenGL buffers
  loadResources :: IO ()
  loadResources = do
    loadTexture "background.png" 0
    return ()
    
  -- | Renders the current state.
  updateDisplay :: IORef TitleState -> IO ()
  updateDisplay stateVar = do
    clear [ColorBuffer]
    drawTitle stateVar
    -- when (slowKey == Press) drawSlowMotion
    -- forM_ (M.assocs $ stShapes state) (fst . snd) -- Draw each one
    swapBuffers
  
  -- | Draw the Silkworm title screen
  drawTitle :: IORef TitleState -> IO ()
  drawTitle stateVar = preservingMatrix $ do
    -- matrixMode $= Modelview 0
    -- loadIdentity
    -- forM_ (tsObjects state) $ \(TitleObject {tImage = image}) -> do
    --   
    -- renderTexture 0 (-400) (-300) 800 600
    state <- get stateVar
    stateVar $= state { tsAngle = ((tsAngle state) + 5.0) }
    drawTunnel testTunnel (tsAngle state)
    
    -- seed <- newStdGen
    -- let (s1, s2) = split seed
    --     bump1 = (randomBumpyRaster rasterRect 40 10.0 s1)
    --     bump2 = (randomBumpyRaster rasterRect 20 20.0 s2)
    --     composite = ((testTunnel #+ 1) #*# (bump1 #* 0.5) #*# (bump2 #* 0.5))
    --   in drawTunnel testTunnel
      
    -- let font = (tsFont state)
    -- font <- createTextureFont "shrewsbury.ttf"
    -- setFontFaceSize font 72 72
    -- renderFont font "Silkworm!" All
    
    -- scale 3.75 3.75 (1.0 :: GLfloat)
    let render str = do
          translate (Vector3 0.0 0.0 (0.0 :: GLfloat))
          renderString Fixed8x16 str
    
    color $ Color3 1.0 1.0 (1.0 :: GLfloat)
    render "Silkworm!"
    