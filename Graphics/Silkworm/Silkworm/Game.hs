module Silkworm.Game where
  
  -- General-purpose Modules
  import Data.Array as A
  import Data.Array ((!))
  import qualified Data.Map as M
  import Data.IORef (IORef, newIORef)
  import Control.Monad (forM, forM_, replicateM, replicateM_, foldM, foldM_, when)
  import System.Directory (getCurrentDirectory)
  
  -- Physics Modules
  import qualified Physics.Hipmunk as H
  
  -- Graphics Modules
  import Graphics.UI.GLFW (
    Key(..), KeyButtonState(..), SpecialKey(..), getKey,
    BitmapFont(..), renderString,
    mousePos, windowSize,
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
    Position(..), Size(..),
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
  import Silkworm.HipmunkHelper (newSpaceWithGravity, newStaticLine)
  import Silkworm.WaveFront (readWaveFront)
  import Silkworm.Object3D (Object3D(..), deform, faces)
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
    -- gsLevel     :: Object3D,
    gsResources :: [FilePath],
    gsActives   :: [GameObject],
    gsInactives :: [GameObject]
  } deriving Show
  
  -- level1 = array ((0, 0), (1, 1)) [((0, 0), 0.0), ((0, 1), 5.0), ((1, 0), 10.0), ((1, 1), 20.0)] :: Array (Int, Int) Double
  level1 = rasterizeLines [((0, 8), (20, 12))] ((0, 0), (20, 20)) 5.0
  
  -------------------------------------------------------
  -- Game Functions
  -------------------------------------------------------
  
  -- | Return a GameState object set to reasonable initial values
  newGameState :: IO GameState
  newGameState = do
    space <- newSpaceWithGravity
    cwd <- getCurrentDirectory
    ground <- newStaticLine space (H.Vector (-0.5) (-0.8)) (H.Vector (0.5) (-0.8))
    worm <- createWorm
    -- level <- return $ maskToObject3D level1
    return (GameState 0.0 space [cwd] [worm] [])
  
  generateRandomGround :: H.Space -> IO ()
  generateRandomGround space =
    return ()
  
  startGame :: IO ()
  startGame = do
    stateRef <- newGameState >>= newIORef
    -- mouseButtonCallback   $= processMouseInput stateRef
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
    -- drawTunnel tunnel 0.0
    Position cx cy <- get mousePos
    Size w h <- get windowSize
    -- putStrLn ((show cx) ++ " " ++ (show cy) ++ " " ++ (show w) ++ " " ++ (show h))
    preservingMatrix $ do
      translate (Vector3 (1.5 -(fromIntegral cx) / (fromIntegral w) * 4)
                         (-1.5 +(fromIntegral cy) / (fromIntegral h) * 4) (0 :: GLfloat))
      -- drawLevel (gsLevel state)
        
      preservingMatrix $ do
        translate (Vector3 (-0.5) (-0.5) (-3 :: GLfloat))
        rotate (gsAngle state) (Vector3 0 1 0 :: Vector3 GLfloat)
        -- scale 4.0 4.0 (4.0 :: Float)
        forM_ (gsActives state) drawGameObject
    angle <- return (gsAngle state)
    stateRef $= state { gsAngle = angle + 1.0 }
    swapBuffers
  

  type DepthMask = A.Array (Int, Int) Double

  innerBounds ary = ((a + 1, b + 1), (c - 1, d - 1))
    where ((a, b), (c, d)) = bounds ary
  
  -- maskToObject3D :: DepthMask -> Object3D
  -- maskToObject3D a = Object3D "tunnel" (map face (range (innerBounds a)))
  --   where
  --     -- size = fromIntegral $ (snd . snd . bounds) a
  --     boxpts      (x, y) = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
  --     orthogonals (x, y) = [(x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y)]
  --     point     i@(x, y) = (fromIntegral x, fromIntegral y, a ! i)
  --     face      i@(x, y) = zip (map point (boxpts i))
  --                              (map normal (boxpts i))
  --     
  --     normal   i@(x, y) | inRange (innerBounds a) i =
  --                         let center = point i
  --                             others = map ((center `minus`) . point) (orthogonals i)
  --                         in (0, 0, 1) `plus` (average (map (cross center) others))
  --                       | otherwise                         = (0, 0, -1)
  --     -- Auxiliary
  --     average [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3), (a4, b4, c4)] =
  --              ((a1 + a2 + a3 + a4) / 4, (b1 + b2 + b3 + b4) / 4, (c1 + c2 + c3 + c4) / 4)
  --     average _ = error "expects 4 values in array"-- hack case for now
  --     minus (a, b, c) (x, y, z) = (a - x, b - y, c - z)
  --     plus (a, b, c) (x, y, z) = (a + x, b + y, c + z)
  --     -- shrink n = (fromIntegral n) / size
  
  drawLevel :: Object3D -> IO ()
  drawLevel level = do
    preservingMatrix $ do
      translate (Vector3 (-2) (-2.4) (-4 :: GLfloat))
      -- translate (Vector3 (0) (0) (-4 :: GLfloat))
      scale (0.4) (0.4) ((0.4) :: GLfloat)
      drawObject $ level
      return ()
  
  drawObject :: Object3D -> IO ()
  drawObject obj = do
    forM_ (faces obj) $ \face -> renderPrimitive Polygon $ do
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
  
  createWorm :: IO GameObject
  createWorm = do
    f <- readFile "silkworm.obj"
    o <- return $ readWaveFront f
    let draw = drawObject o
    return GameObject { gDraw = Action "draw silkworm" draw }
  
  -- processMouseInput :: IORef State -> MouseButton -> KeyButtonState -> IO ()
  -- processMouseInput _        _   Press   =
  -- processMouseInput stateVar btn Release = do
  --   rotateKeyCCW <- getKey (SpecialKey LSHIFT)
  --   rotateKeyCW  <- getKey (SpecialKey RSHIFT)
  --   let angVel = case (rotateKeyCCW, rotateKeyCW) of
  --                  (Press,   Release) -> 50
  --                  (Release, Press)   -> (-50)
  --                  _                  -> 0
  --   (shape,add,draw,remove) <- (case btn of
  --     ButtonLeft  -> createCircle
  --     ButtonRight -> createSquare
  --     _           -> createTriPendulum) angVel
  -- 
  --   state <- get stateVar
  --   let space = stSpace state
  --   add space >> stateVar $= state {
  --     stShapes = M.insert shape (draw, remove space) $ stShapes state}
  
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
