module Silkworm.Title where
  
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
    PrimitiveMode(..), renderPrimitive,
    preservingMatrix)
  import Graphics.Rendering.OpenGL.GL.StateVar (get)
  import Graphics.Rendering.FTGL (
    RenderMode(..), Font, createTextureFont, createPolygonFont,
    setFontFaceSize, renderFont, getFontError)
  import Silkworm.WindowHelper (getPressedKeys, keyIsPressed)
  import Silkworm.ImageHelper (loadTexture, renderTexture)
  import Silkworm.LevelGenerator (rasterizeLines)
  
  type KeyMap = Map.Map Key Bool
  
  data TitleObject = TitleObject {
    tPosition   :: (GLfloat, GLfloat),
    tPoints     :: [(GLfloat, GLfloat)],
    tRadians    :: Float,
    tTextureId  :: Integer
  }
  
  data TitleState = TitleState {
    tsObjects :: [TitleObject],
    tsFont    :: Font
  }
  
  testTunnel = rasterizeLines [((50,50), (100,130)), ((0,100), (200,100))] ((0, 0), (200, 200)) 20.0
  
  drawTunnel :: Array.Array (Int, Int) Float -> (Int, Int) -> IO ()
  drawTunnel t (px, py) = preservingMatrix $ do
    translate (Vector3 (-100) (-100) (0.0 :: GLfloat))
    let point = renderPrimitive Points . vertex in
      forM_ (Array.assocs t) $ \((x, y), d) -> do
        color $ Color3 (d / 20) 0.0 (0.0 :: GLfloat)
        point $ Vertex3 (fromIntegral (x + px)) (fromIntegral (y + py)) (0.0 :: GLfloat)
    
  -- | Return a TitleState object set to reasonable initial values
  newTitleState :: IO TitleState
  newTitleState = do
    -- bgImage <- readImage "background.png"
    -- let objs = [TitleObject (0, 0) [] 0.0 bgImage]
    let objs = []
    font <- createPolygonFont "scribbled.ttf"
    -- putStrLn (show $ getFontError font)
    -- exitWith (ExitFailure (-1))
    return (TitleState objs font)
  
  showTitleScreen :: IO ()
  showTitleScreen = do
    loadResources
    state <- newTitleState
    titleScreenLoop state
  
  titleScreenLoop :: TitleState -> IO ()
  titleScreenLoop state = do
    keySet <- getPressedKeys [SpecialKey ESC]
    updateDisplay state
    if keyIsPressed (SpecialKey ESC) keySet
      then return ()
      else titleScreenLoop state
  
  -- | Load title screen textures into OpenGL buffers
  loadResources :: IO ()
  loadResources = do
    loadTexture "background.png" 0
    return ()
    
  -- | Renders the current state.
  updateDisplay :: TitleState -> IO ()
  updateDisplay state = do
    clear [ColorBuffer]
    drawTitle state
    -- when (slowKey == Press) drawSlowMotion
    -- forM_ (M.assocs $ stShapes state) (fst . snd) -- Draw each one
    swapBuffers
  
  -- | Draw the Silkworm title screen
  drawTitle :: TitleState -> IO ()
  drawTitle state = preservingMatrix $ do
    -- forM_ (tsObjects state) $ \(TitleObject {tImage = image}) -> do
    --   
    renderTexture 0 (-400) (-300) 800 600
    
    drawTunnel testTunnel (40, 40)
    -- let font = (tsFont state)
    -- font <- createTextureFont "shrewsbury.ttf"
    -- setFontFaceSize font 72 72
    -- renderFont font "Silkworm!" All
    
    -- scale 3.75 3.75 (1.0 :: GLfloat)
    -- let render str = do
    --       translate (Vector3 0.0 (-16) (0.0 :: GLfloat))
    --       renderString Fixed8x16 str
    -- 
    -- color $ Color3 1.0 1.0 (1.0 :: GLfloat)
    -- render "Silkworm!"
    