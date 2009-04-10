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
  import Graphics.UI.GLFW
  import Graphics.Rendering.OpenGL
  
  -- Silkworm-specific Modules
  import Silkworm.Action
  import Silkworm.Behavior
  import Silkworm.Constants
  import Silkworm.GameObject
  import Silkworm.HipmunkHelper
  import Silkworm.LevelGenerator
  import Silkworm.Math
  import Silkworm.OpenGLHelper
  import Silkworm.WaveFront
  import Silkworm.Mesh
  import Silkworm.Substance
  
  -------------------------------------------------------
  -- Game Data and Type Declarations
  -------------------------------------------------------
  
  
  -- The Game State : Keep track of the Chipmunk engine state as well as our game
  -- objects, both active (on-screen) and inactive (off-screen)
  data GameState = GameState {
    gsAngle     :: Float,
    gsSpace     :: H.Space,
    -- gsLevel     :: Mesh,
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
    space <- newSpaceWithGravity gravity
    cwd <- getCurrentDirectory
    ground <- makeWall (H.Vector (-1) (-0.5)) (H.Vector (1) (-0.5))
    space `includes` ground
    square <- makeCrate 0.4 (H.Vector 0 3)
    space `includes` square
    -- worm <- createWorm space
    -- sign <- createSign space
    -- level <- return $ maskToMesh level1
    return (GameState (0.0) space [cwd] [ground, square] [])
  
  generateRandomGround :: H.Space -> IO ()
  generateRandomGround space =
    return ()
  
  startGame :: IO ()
  startGame = do
    
    configureProjection (Perspective 90.0) Nothing
    -- configureProjection Orthogonal Nothing
    
    stateRef <- newGameState >>= newIORef
    
    -- mouseButtonCallback   $= processMouseInput stateRef
    windowSizeCallback $= resizeWindow
    
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
    
    (x, y) <- getMousePan
    
    lookingAt (Vector3 x y 2.5) (Vector3 0 0 0) (Vector3 0 1 0) $ do
      moveLight $ Vector3 0.5 0.5 (-0.8)
      -- drawLevel (gsLevel state)
      forM_ (gsActives state) drawGameObject
    
    -- angle <- return (gsAngle state)
    -- stateRef $= state { gsAngle = angle + 0.01 }
    swapBuffers
  
  getMousePan :: IO (Double, Double)
  getMousePan = do
    Position cx cy <- get mousePos
    Size w h <- get windowSize
    
    let x = (fromIntegral (cx - (w `div` 2))) / (fromIntegral w) * (-1)
        y = (fromIntegral (cy - (h `div` 2))) / (fromIntegral h) * (1)
    
    return (x, y)
  
  type DepthMask = A.Array (Int, Int) Double

  innerBounds ary = ((a + 1, b + 1), (c - 1, d - 1))
    where ((a, b), (c, d)) = bounds ary
  
  -- maskToMesh :: DepthMask -> Mesh
  -- maskToMesh a = Mesh "tunnel" (map face (range (innerBounds a)))
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
  
  drawLevel :: Mesh -> IO ()
  drawLevel level = do
    preservingMatrix $ do
      translate (Vector3 (-2) (-2.4) (-4 :: GLfloat))
      -- translate (Vector3 (0) (0) (-4 :: GLfloat))
      scale (0.4) (0.4) ((0.4) :: GLfloat)
      drawObject $ level
      return ()
  
  drawObject :: Mesh -> IO ()
  drawObject obj = do
    forM_ (faces obj) $ \face -> renderPrimitive Polygon $ do
      forM_ face $ \((vx, vy, vz), (nx, ny, nz)) -> do
        normal $ Normal3 nx ny nz
        vertex $ Vertex3 vx vy vz
  
  wormSegments = 10 :: Integer
  
  -- createWorm :: H.Space -> IO GameObject
  -- createWorm space = do
  --   fileData <- readFile "silkworm.obj"
  --   wormObj <- return $ readWaveFront fileData
  --   
  --   -- let ctrls = crSplineN wormSegments [(-2,0,0),(-1,0,0),(0,0.5,0),(1,0,0),(2,0,0)]
  --   let morph@Morph{ mphControls = m_ctrls } = mkMorph wormObj wormSegments
  --       segs = zip [0..] m_ctrls
  --   
  --   -- Generate the worm body parts
  --   shapes <- forM segs $ \(i, (x, y, z)) -> do
  --     shape <- newShape (H.Circle 0.2) 2.0 0.2 0.0
  --     includeShape space shape
  --     H.setPosition (H.getBody shape) (H.Vector ((realToFrac x) + 1.5) ((realToFrac y) + 2.5))
  --     H.setVelocity (H.getBody shape) (H.Vector 0 ((fromIntegral i)/10))
  --     return shape
  --   
  --   -- Create pivots between body parts
  --   let pairs = zip (drop 0 shapes) (drop 1 shapes)
  --   forM_ pairs $ \(s1, s2) -> do
  --     p1 <- H.getPosition (H.getBody s1)
  --     p2 <- H.getPosition (H.getBody s2)
  --     j <- H.newJoint (H.getBody s1) (H.getBody s2) (H.Pivot (midpoint p1 p2))
  --     H.spaceAdd space j
  --   
  --   let draw = do
  --         -- putStrLn "draw silkworm"
  --         ctrls <- forM shapes $ \s -> do
  --           (H.Vector x y) <- H.getPosition (H.getBody s)
  --           return ((realToFrac x), (realToFrac y), 0)
  --         
  --         preservingMatrix $ do
  --           let (x, y, z) = center ctrls
  --           translate $ Vector3 x (y-0) 0
  --           color  $ Color3 0.2 0.7 (0.2 :: GLfloat)
  --           drawObject (blockMorph morph ctrls)
  --   
  --   mainShape <- return (head shapes)
  --   return $ (gameObject mainShape) { gDraw = Action "draw silkworm" draw
  --                                     , gMorph = morph }
  -- 
  toDegrees rad = rad / pi * 180
  
  orientationForShape shape action = do
    (H.Vector x y) <- H.getPosition $ H.getBody shape
    angle          <- H.getAngle    $ H.getBody shape
    -- putStrLn $ "x: " ++ (show x) ++ ", y: " ++ (show y)
    preservingMatrix $ do
      translate    $ Vector3 x y 0
      rotate (toDegrees angle) $ Vector3 0 0 (1 :: GLfloat)
      action
    
  ------------------------------------------------------------
  -- Simulation bookkeeping
  ------------------------------------------------------------

  -- | Advances the time.
  advanceTime :: IORef GameState -> Double -> KeyButtonState -> IO Double
  advanceTime stateRef oldTime slowKey = do
    newTime <- get time
  
    -- Advance simulation
    let slower         = if slowKey == Press then slowdown else 1
        mult           = frameSteps / (framePeriod * slower)
        framesPassed   = truncate $ mult * (newTime - oldTime)
        simulNewTime   = oldTime + toEnum framesPassed / mult
    advanceSimulTime stateRef $ min maxSteps framesPassed
  
    -- Correlate with reality
    newTime' <- get time
    let diff = newTime' - simulNewTime
        sleepTime = ((framePeriod * slower) - diff) / slower
    when (sleepTime > 0) $ sleep sleepTime
    return simulNewTime
    where
      -- | Advances the time in a certain number of steps.
      advanceSimulTime :: IORef GameState -> Int -> IO ()
      advanceSimulTime _        0     = return ()
      advanceSimulTime stateRef steps = do
        state <- get stateRef
        replicateM_ steps $ H.step (gsSpace state) frameDelta
      