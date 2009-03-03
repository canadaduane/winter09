module Silkworm.Init (State(..), initializeGame) where
  
  import Data.IORef (IORef, newIORef)
  import Data.Map (Map, assocs, fromList)
  import System.Exit
  import Graphics.UI.GLFW (WindowMode(..), initialize, openWindow, windowTitle, windowCloseCallback)
  import Graphics.Rendering.OpenGL
  import Physics.Hipmunk (initChipmunk, Shape, Space, newSpace, setGravity)
  import Silkworm.Constants
  import Silkworm.Misc
  
  data State = State {
    stSpace  :: Space,
    stShapes :: Map Shape (
      IO () {- Drawing -},
      IO () {- Removal -}
    )}
  
  initialState :: IO State
  initialState = do
    space  <- newSpace
    setGravity space gravity
    
    generateRandomGround space
    
    return (State space shapes)
    
    where
      shapes = fromList []
  
  generateRandomGround :: Space -> IO ()
  generateRandomGround space =
    return ()
  
  initializeGame :: IO (IORef State)
  initializeGame = do
    -- Start physics engine
    initChipmunk
    
    -- Initialize state reference variable
    stateRef <- initialState >>= newIORef
    
    -- Open a window using GLFW
    assertTrue initialize "Failed to init GLFW"
    let size = (uncurry Size) windowDimensions
    assertTrue (openWindow size [] Window) "Failed to open a window"
    windowTitle $= "Silkworm"
    
    -- Initialize OpenGL
    clearColor  $= Color4 1 1 1 1
    pointSmooth $= Enabled
    pointSize   $= 3
    lineSmooth  $= Enabled
    lineWidth   $= 2.5
    blend       $= Enabled
    blendFunc   $= (SrcAlpha, OneMinusSrcAlpha)
    matrixMode  $= Projection
    loadIdentity
    ortho (-320) 320 (-240) 240 (-1) 1
    translate (Vector3 0.5 0.5 0 :: Vector3 GLfloat)
    
    -- Close window nicely
    windowCloseCallback   $= exitWith ExitSuccess
    
    return stateRef
    