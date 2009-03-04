module Silkworm.Game (State(..), initializeGame) where

  -- General-purpose Modules
  import Data.IORef (IORef, newIORef)

  import qualified Data.Map (
    Map,
    assocs,
    fromList
  ) as Map
  import qualified System.Exit (
    ExitCode(..),
    exitWith
  ) as Exit
  -- import Control.Monad
  
  import qualified Graphics.Rendering.OpenGL as GL
  
  -- Physics Modules
  import Physics.Hipmunk (
    -- Global Functions
    initChipmunk,
    
    -- Shape Functions
    Shape,
  
    -- Space Functions
    Space,
    newSpace,
    setGravity
  ) as Hipmunk

  -- Silkworm-specific Modules
  -- import Silkworm.Constants
  -- import Silkworm.Drawing
  -- import Silkworm.Misc
  import Silkworm.WindowHelper (initWindow)
  import Silkworm.OpenGLHelper (initOpenGL)
  import Silkworm.HipmunkHelper (newSpaceWithGravity)
  
  -------------------------------------------------------
  -- Game Data and Type Declarations
  -------------------------------------------------------
  
  -- Various behaviors of the GameObjects
  data Behavior = BeStationary
                | BeSimple
                | BeCrazy
  instance Show Behavior where
    show BeStationary = "Stationary Behavior"
    show BeSimple     = "Simple Behavior"
    show BeCrazy      = "Crazy Behavior"
  
  -- Named IO Actions (for debugging and display)
  data Action = Action String (IO ())
  instance Show Action where
    show (Action name action) = name
  
  -- Special action that wraps a function in an "Anonymous" Action
  anonymAction :: IO () -> Action
  anonymAction fn = Action "Anonymous" fn
  
  -- Special action that does nothing (used for default values and tests)
  inaction :: Action
  inaction = Action "Inaction" nothing
    where nothing = do return ()
  
  -- The Game Object : Here is where we define everything to do with each moving thing
  data GameObject = GameObject {
    gShape    :: Shape,
    gBehavior :: Behavior,
    gDraw     :: Action,
    gRemove   :: Action
  } deriving Show
  
  -- The Game State : Keep track of the Chipmunk engine state as well as our game
  -- objects, both active (on-screen) and inactive (off-screen)
  data GameState = GameState {
    gsSpace     :: Space,
    gsActives   :: [GameObject],
    gsInactives :: [GameObject]
  } deriving Show
  
  -------------------------------------------------------
  -- Game Functions
  -------------------------------------------------------
  
  -- | Return a GameState object set to reasonable initial values
  newGameState :: IO GameState
  newGameState =
    let objs = []
    do space <- initialSpaceWithGravity
       return (GameState space objs)
  
  generateRandomGround :: Space -> IO ()
  generateRandomGround space =
    return ()
  
  startGameLoop :: IORef State -> IO ()
  startGameLoop stateRef = do
    -- Let's go!
    now <- get time
    gameLoop stateRef now

  -- | The simulation loop.
  gameLoop :: IORef State -> Double -> IO ()
  gameLoop stateRef oldTime = do
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
    updateDisplay stateRef slowKey
    newTime <- advanceTime stateRef oldTime slowKey
  
    -- Continue the game as long as quit signal has not been given
    when (quitKey != Press)
      (gameLoop stateRef newTime)

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
