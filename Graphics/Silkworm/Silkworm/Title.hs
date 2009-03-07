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
    renderTexture 0 (-1) (-1) 2 2
    
    state <- get stateVar
    stateVar $= state { tsAngle = ((tsAngle state) + 5.0) }
    -- drawTunnel testTunnel (tsAngle state)
    
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
    
    scale (1.0/800*3) (1.0/600*3) (1.0 :: GLfloat)
    let render str = do
          translate (Vector3 0.0 0.0 (0.1 :: GLfloat))
          renderString Fixed8x16 str
    
    color $ Color3 1.0 0.0 (0.0 :: GLfloat)
    -- normal $ Normal3 0 0 (-1.0 :: GLfloat)
    render "Silkworm!"
    