module Silkworm.Title where
  
  import qualified Data.Map as Map
  import Data.IORef (IORef, newIORef)
  import Graphics.UI.GLFW (
    Key(..), SpecialKey(..),
    BitmapFont(..), renderString,
    swapBuffers)
  import Graphics.Rendering.OpenGL (
    GLint, GLfloat,
    ClearBuffer(..), clear,
    Vector3(..), scale, translate, rotate,
    Color3(..), color,
    preservingMatrix)
  import Graphics.Rendering.OpenGL.GL.StateVar (get)
  import Silkworm.WindowHelper (getPressedKeys, keyIsPressed)
  
  type KeyMap = Map.Map Key Bool
  
  data TitleObject = TitleObject {
    tPosition   :: (GLfloat, GLfloat),
    tPoints     :: [(GLfloat, GLfloat)],
    tRadians    :: Float
  }
  
  data TitleState = TitleState {
    tsObjects :: [TitleObject]
  }

  -- | Return a TitleState object set to reasonable initial values
  newTitleState :: IO TitleState
  newTitleState = do
    let objs = []
    return (TitleState objs)
  
  showTitleScreen :: IO ()
  showTitleScreen = do
    tsStateVar <- newTitleState >>= newIORef
    titleScreenLoop tsStateVar

  titleScreenLoop :: IORef TitleState -> IO ()
  titleScreenLoop tsStateVar = do
    keySet <- getPressedKeys [SpecialKey ESC]
    updateDisplay tsStateVar
    if keyIsPressed (SpecialKey ESC) keySet
      then return ()
      else titleScreenLoop tsStateVar

  -- | Renders the current state.
  updateDisplay :: IORef TitleState -> IO ()
  updateDisplay tsStateVar = do
    state <- get tsStateVar
    clear [ColorBuffer]
    drawTitle
    -- when (slowKey == Press) drawSlowMotion
    -- forM_ (M.assocs $ stShapes state) (fst . snd) -- Draw each one
    swapBuffers

  drawTitle :: IO ()
  drawTitle = preservingMatrix $ do
    scale 0.75 0.75 (1.0 :: GLfloat)
    let render str = do
          translate (Vector3 0.0 (-16) (0.0 :: GLfloat))
          renderString Fixed8x16 str

    color $ Color3 0.0 0.0 (1.0 :: GLfloat)
    render "Silkworm!"
    