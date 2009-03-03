import Control.Monad
import Data.IORef
import qualified Data.Map as Map
import System.Exit

import Graphics.UI.GLFW as Win
import Graphics.Rendering.OpenGL as GL
import qualified Physics.Hipmunk as Physics


data State = State {
  stSpace  :: Physics.Space,
  stShapes :: Map.Map Physics.Shape (
    IO () {- Drawing -},
    IO () {- Removal -}
  )}

main :: IO ()
main = do
  space <- initPhysics
  state <- newIORef (State space (Map.fromList []))
  initWindow
  
  -- Let's go!
  now <- get time
  loop state now
  -- return ()

initWindow :: IO ()
initWindow = do
  assertTrue Win.initialize "Failed to initialize window library"
  assertTrue (openWindow (Size 800 600) [] Window) "Failed to open a window"
  Win.windowTitle $= "Silk Worm"

initPhysics :: IO Physics.Space
initPhysics = do
  Physics.initChipmunk
  space <- Physics.newSpace
  Physics.setElasticIterations space 10
  Physics.setGravity space downVector
  return space
  
  where
    downVector = Physics.Vector 0 (-230)
  

-- | The game loop.
loop :: IORef State -> Double -> IO ()
loop stateVar oldTime = do
  -- Some key states
  slowKey  <- getKey (SpecialKey ENTER)
  quitKey  <- getKey (SpecialKey ESC)
  clearKey <- getKey (SpecialKey DEL)

  -- Quit?
  when (quitKey == Press) (terminate >> exitWith ExitSuccess)

  -- Clear?
  -- when (clearKey == Press) $ do
  --   destroyState =<< readIORef stateVar
  --   initialState >>= writeIORef stateVar

  -- Update display and time
  -- updateDisplay stateVar slowKey
  -- newTime <- advanceTime stateVar oldTime slowKey
  sleep 10
  newTime <- get time
  loop stateVar newTime


