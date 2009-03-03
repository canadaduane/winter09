module Silkworm.Game where
  
  import Graphics.UI.GLFW
  import Data.IORef
  import Silkworm.Constants
  import Silkworm.Drawing
  import Silkworm.Misc
  
  startGame :: IORef State -> IO ()
  startGame stateRef = do
    -- Let's go!
    now <- get time
    loop stateRef now
  
  -- | The simulation loop.
  loop :: IORef State -> Double -> IO ()
  loop stateVar oldTime = do
    -- Some key states
    slowKey  <- getKey (SpecialKey ENTER)
    quitKey  <- getKey (SpecialKey ESC)
    clearKey <- getKey (SpecialKey DEL)
    
    -- Quit?
    -- when (quitKey == Press) (terminate >> exitWith ExitSuccess)
    
    -- Clear?
    -- when (clearKey == Press) $ do
    --   destroyState =<< readIORef stateVar
    --   initialState >>= writeIORef stateVar
    
    -- Update display and time
    -- updateDisplay stateVar slowKey
    -- newTime <- advanceTime stateVar oldTime slowKey
    newTime <- get time
    loop stateVar newTime
  