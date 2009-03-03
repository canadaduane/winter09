module Silkworm.Game where
  
  import Control.Monad
  import Graphics.UI.GLFW
  import Graphics.Rendering.OpenGL
  -- import Graphics.Rendering.OpenGL.GL.StateVar (StateVar, get, ($=))
  import Data.IORef
  import qualified Data.Map as M
  import qualified Physics.Hipmunk as H
  import Silkworm.Init (State(..))
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
  loop stateRef oldTime = do
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
    updateDisplay stateRef slowKey
    newTime <- advanceTime stateRef oldTime slowKey
    loop stateRef newTime
  
  -- | Renders the current state.
  updateDisplay :: IORef State -> KeyButtonState -> IO ()
  updateDisplay stateRef slowKey  = do
    state <- get stateRef
    clear [ColorBuffer]
    -- drawInstructions
    -- when (slowKey == Press) drawSlowMotion
    forM_ (M.assocs $ stShapes state) (fst . snd) -- Draw each one
    swapBuffers

  ------------------------------------------------------------
  -- Simulation bookkeeping
  ------------------------------------------------------------

  -- | Advances the time in a certain number of steps.
  advanceSimulTime :: IORef State -> Int -> IO ()
  advanceSimulTime _        0     = return ()
  advanceSimulTime stateRef steps = do
    removeOutOfSight stateRef
    state <- get stateRef
    replicateM_ steps $ H.step (stSpace state) frameDelta

  -- | Removes all shapes that may be out of sight forever.
  removeOutOfSight :: IORef State -> IO ()
  removeOutOfSight stateRef = do
    state   <- get stateRef
    shapes' <- foldM f (stShapes state) $ M.assocs (stShapes state)
    stateRef $= state {stShapes = shapes'}
      where
        f shapes (shape, (_,remove)) = do
          H.Vector x y <- H.getPosition $ H.getBody shape
          if y < (-350) || abs x > 800
            then remove >> return (M.delete shape shapes)
            else return shapes
  
  -- | Advances the time.
  advanceTime :: IORef State -> Double -> KeyButtonState -> IO Double
  advanceTime stateRef oldTime slowKey = do
    newTime <- get time
    
    -- Advance simulation
    let slower = if slowKey == Press then slowdown else 1
        mult   = frameSteps / (framePeriod * slower)
        framesPassed   = truncate $ mult * (newTime - oldTime)
        simulNewTime   = oldTime + toEnum framesPassed / mult
    advanceSimulTime stateRef $ min maxSteps framesPassed
    
    -- Correlate with reality
    newTime' <- get time
    let diff = newTime' - simulNewTime
        sleepTime = ((framePeriod * slower) - diff) / slower
    when (sleepTime > 0) $ sleep sleepTime
    return simulNewTime
