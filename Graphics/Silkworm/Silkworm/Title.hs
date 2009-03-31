module Silkworm.Title where
  
  import System.Random (newStdGen, split)
  import System.Directory (getCurrentDirectory)
  import System.Exit (ExitCode(..), exitWith)
  import Control.Monad (forM, forM_, when)
  import qualified Data.Map as Map
  import Data.IORef (IORef, newIORef)
  import Data.Array.IArray as Array
  import Graphics.UI.GLFW (
    Key(..), SpecialKey(..),
    BitmapFont(..), renderString,
    sleep, swapBuffers)
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
  import Silkworm.Game (startGame)
  
  type KeyMap = Map.Map Key Bool
  
  data TitleObject = TitleObject {
    tPosition   :: (GLfloat, GLfloat),
    tPoints     :: [(GLfloat, GLfloat)],
    tRadians    :: Float,
    tTextureId  :: Integer
  }
  
  data MenuSelection = NoSelection | StartGame | QuitGame
    deriving Eq
  
  data TitleState = TitleState {
    tsObjects :: [TitleObject],
    tsHover   :: MenuSelection,
    tsSelect  :: MenuSelection
  }
  
  -- | Return a TitleState object set to reasonable initial values
  newTitleState :: IO TitleState
  newTitleState = do
    -- bgImage <- readImage "background.png"
    -- let objs = [TitleObject (0, 0) [] 0.0 bgImage]
    let objs = []
    -- putStrLn (show $ getFontError font)
    -- exitWith (ExitFailure (-1))
    return (TitleState objs StartGame NoSelection)
  
  showTitleScreen :: IO ()
  showTitleScreen = do
    loadResources
    state <- newTitleState >>= newIORef
    titleScreenLoop state
  
  titleScreenLoop :: IORef TitleState -> IO ()
  titleScreenLoop stateVar = do
    state <- get stateVar
    keySet <- getPressedKeys [SpecialKey ENTER, SpecialKey UP, SpecialKey DOWN]
    updateDisplay stateVar
    
    when (keyIsPressed (SpecialKey UP) keySet) $ do
      stateVar $= state { tsHover = StartGame }
    when (keyIsPressed (SpecialKey DOWN) keySet) $ do
      stateVar $= state { tsHover = QuitGame }
    when (keyIsPressed (SpecialKey ENTER) keySet) $ do
      stateVar $= state { tsSelect = (tsHover state) }
    
    case tsSelect state of
      NoSelection -> titleScreenLoop stateVar
      StartGame   -> startGame
      QuitGame    -> exitWith ExitSuccess
  
  -- | Load title screen textures into OpenGL buffers
  loadResources :: IO ()
  loadResources = do
    loadTexture "background.png" 0
    return ()
    
  -- | Renders the current state.
  updateDisplay :: IORef TitleState -> IO ()
  updateDisplay stateVar = do
    clear [ColorBuffer, DepthBuffer]
    drawTitle stateVar
    -- when (slowKey == Press) drawSlowMotion
    -- forM_ (M.assocs $ stShapes state) (fst . snd) -- Draw each one
    swapBuffers
    sleep 0.050
  
  drawText :: Float -> Float -> Float -> String -> IO ()
  drawText x y s str = preservingMatrix $ do
    translate (Vector3 x y (-2))
    scale (1.0/800*s) (1.0/600*s) 1.0
    renderString Fixed8x16 str
  
  -- | Draw the Silkworm title screen
  drawTitle :: IORef TitleState -> IO ()
  drawTitle stateVar = do
    state <- get stateVar
    -- stateVar $= state { tsAngle = ((tsAngle state) + 5.0) }

    preservingMatrix $ do
      translate (Vector3 0 0 (-2) :: Vector3 Float)
      renderTexture 0 (-1) (-1) 2 2
      
      -- let font = (tsFont state)
      -- font <- createTextureFont "shrewsbury.ttf"
      -- setFontFaceSize font 72 72
      -- renderFont font "Silkworm!" All
    
    color $ Color3 0.0 0.0 (0.0 :: GLfloat)
    drawText (-0.2) (0.5) 6 "Silkworm!"

    color $ Color3 1.0 0.2 (0.2 :: GLfloat)
    drawText (-0.5) (-0.0) (selectionSize state StartGame) "Start Game"
    drawText (-0.5) (-0.1) (selectionSize state QuitGame) "Quit"
    
    where
      selectionSize state s =
        if s == (tsHover state)
          then 6
          else 5
    