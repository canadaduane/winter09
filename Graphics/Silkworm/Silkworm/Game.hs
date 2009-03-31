module Silkworm.Game (startGame) where
  
  -- General-purpose Modules
  import Data.Array as A
  import Data.Map as M
  import Data.IORef (IORef, newIORef)
  import Control.Monad (forM, forM_, replicateM, replicateM_, foldM, foldM_, when)
  import System.Directory (getCurrentDirectory)
  
  -- Physics Modules
  import qualified Physics.Hipmunk as H
  
  -- Graphics Modules
  import Graphics.UI.GLFW (
    Key(..), KeyButtonState(..), SpecialKey(..), getKey,
    BitmapFont(..), renderString,
    time, sleep, swapBuffers)
  
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
  
  -- Silkworm-specific Modules
  import Silkworm.Constants (slowdown, frameDelta, frameSteps, framePeriod, maxSteps)
  -- import Silkworm.Drawing
  -- import Silkworm.Misc
  -- import Silkworm.WindowHelper (initWindow)
  -- import Silkworm.OpenGLHelper (initOpenGL)
  import Silkworm.LevelGenerator (rasterizeLines)
  import Silkworm.HipmunkHelper (newSpaceWithGravity)
  import Silkworm.WaveFront (readWaveFront, Object3D(..))
  import Silkworm.Math (cross)
  
  -- rasterRect = ((0, 0), (200, 200))
  -- testTunnel = rasterizeLines [((50,50), (100,130)), ((50,100), (150,100))] rasterRect 20.0
  
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
    where nothing = return ()
  
  instance Show H.Shape where
    show s = "Shape"
  instance Show H.Space where
    show s = "Space"
  
  -- The Game Object : Here is where we define everything to do with each moving thing
  data GameObject = GameObject {
    gShape    :: H.Shape,
    gBehavior :: Behavior,
    gDraw     :: Action,
    gRemove   :: Action
  } deriving Show
  
  act :: Action -> IO ()
  act (Action name fn) = do
    -- putStrLn name
    fn
  
  drawGameObject :: GameObject -> IO ()
  drawGameObject obj = act (gDraw obj)
  
  -- The Game State : Keep track of the Chipmunk engine state as well as our game
  -- objects, both active (on-screen) and inactive (off-screen)
  data GameState = GameState {
    gsAngle     :: Float,
    gsSpace     :: H.Space,
    gsResources :: [FilePath],
    gsActives   :: [GameObject],
    gsInactives :: [GameObject]
  } deriving Show
  
  -------------------------------------------------------
  -- Game Functions
  -------------------------------------------------------
  
  -- | Return a GameState object set to reasonable initial values
  newGameState :: IO GameState
  newGameState = do
    space <- newSpaceWithGravity
    cwd <- getCurrentDirectory
    sign <- createSign
    return (GameState 0.0 space [cwd] [sign] [])
  
  generateRandomGround :: H.Space -> IO ()
  generateRandomGround space =
    return ()
  
  startGame :: IO ()
  startGame = do
    stateRef <- newGameState >>= newIORef
    now      <- get time
    gameLoop stateRef now
  
  -- | The main loop
  gameLoop :: IORef GameState -> Double -> IO ()
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
    when (quitKey /= Press)
      (gameLoop stateRef newTime)

  -- | Renders the current state.
  updateDisplay :: IORef GameState -> KeyButtonState -> IO ()
  updateDisplay stateRef slowKey  = do
    state <- get stateRef
    clear [ColorBuffer, DepthBuffer]
    -- let tunnel = rasterizeLines [((0, 0), (70, 70))] ((0, 0), (100, 100)) 15.0
    -- drawTunnel tunnel 0.0

    -- when (slowKey == Press) drawSlowMotion
    -- forM_ (M.assocs $ stShapes state) (fst . snd) -- Draw each one
    state <- get stateRef
    angle <- return (gsAngle state)
    preservingMatrix $ do
      translate (Vector3 (0) (-2) (-3 :: GLfloat))
      rotate angle (Vector3 0 1 0 :: Vector3 GLfloat)
      -- scale 4.0 4.0 (4.0 :: Float)
      forM_ (gsActives state) drawGameObject
    stateRef $= state { gsAngle = angle + 1.0 }
    swapBuffers
  
  level1 = array ((0, 0), (1, 1)) [((0, 0), 0.0), ((0, 1), 5.0), ((1, 0), 10.0), ((1, 1), 20.0)] :: Array (Int, Int) Float
  
  drawTunnel :: A.Array (Int, Int) Float -> Float -> IO ()
  drawTunnel t angle = preservingMatrix $ do
    translate (Vector3 (-1) (-1) (-2 :: GLfloat))
    rotate angle $ Vector3 0.5 1.0 (0.0 :: GLfloat)
    scale 4.0 4.0 (4.0 :: Float)
    color $ Color3 0.5 0.5 (1.0 :: GLfloat) 
    let ((x1, y1), (x2, y2)) = bounds t
        rng = range ((x1, y1), (x2 - 1, y2 - 1))
        shrink n = (fromIntegral n) / 200
        xyz x y = (shrink x, shrink y, (t A.! (x, y)) / 200.0)
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
  
  drawObject :: Object3D -> IO ()
  drawObject (Object3D name faces) = do
    forM_ faces $ \face -> renderPrimitive Polygon $ do
      forM_ face $ \((vx, vy, vz), (nx, ny, nz)) -> do
        normal $ Normal3 nx ny nz
        vertex $ Vertex3 vx vy vz
  
  -- seed <- newStdGen
  -- let (s1, s2) = split seed
  --     bump1 = (randomBumpyRaster rasterRect 40 10.0 s1)
  --     bump2 = (randomBumpyRaster rasterRect 20 20.0 s2)
  --     composite = ((testTunnel #+ 1) #*# (bump1 #* 0.5) #*# (bump2 #* 0.5))
  --   in drawTunnel testTunnel
  
  createSign :: IO GameObject
  createSign = do
    -- let mass   = 20
    --     radius = 20
    --     t = H.Circle radius
    -- b <- H.newBody mass $ H.momentForCircle mass (0, radius) 0
    -- s <- H.newShape b t 0
    -- H.setAngVel b angVel
    -- H.setPosition b =<< getMousePos
    -- H.setFriction s 0.5
    -- H.setElasticity s 0.9
    -- let add space = do
    --       H.spaceAdd space b
    --       H.spaceAdd space s
    -- let draw = do
    --       drawMyShape s t
    -- let remove space = do
    --       H.spaceRemove space b
    --       H.spaceRemove space s
    f <- readFile "sign.obj"
    o <- return $ readWaveFront f
    let draw = drawObject o
    return GameObject { gDraw = Action "draw sign" draw }
  
  ------------------------------------------------------------
  -- Simulation bookkeeping
  ------------------------------------------------------------

  -- | Advances the time in a certain number of steps.
  advanceSimulTime :: IORef GameState -> Int -> IO ()
  advanceSimulTime _        0     = return ()
  advanceSimulTime stateRef steps = do
    removeOutOfSight stateRef
    state <- get stateRef
    replicateM_ steps $ H.step (gsSpace state) frameDelta

  -- | Removes all shapes that may be out of sight forever.
  removeOutOfSight :: IORef GameState -> IO ()
  removeOutOfSight stateRef = do
    return ()
    -- state   <- get stateRef
    -- shapes' <- foldM f (gsShapes state) $ M.assocs (stShapes state)
    -- stateRef $= state {gsShapes = shapes'}
    --   where
    --     f shapes (shape, (_,remove)) = do
    --       H.Vector x y <- H.getPosition $ H.getBody shape
    --       if y < (-350) || abs x > 800
    --         then remove >> return (M.delete shape shapes)
    --         else return shapes

  -- | Advances the time.
  advanceTime :: IORef GameState -> Double -> KeyButtonState -> IO Double
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
