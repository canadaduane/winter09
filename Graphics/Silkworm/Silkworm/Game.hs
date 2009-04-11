module Silkworm.Game where
  
  -- General-purpose Modules
  import Data.Array as A
  import Data.Array ((!))
  import qualified Data.Map as M
  import Data.IORef (IORef, newIORef)
  import Control.Monad (mapM, forM, forM_, replicateM, replicateM_, foldM, foldM_, when)
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
  import Silkworm.ImageHelper
  import Silkworm.LevelGenerator as LG
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
  data GameState = GameState { gsAngle     :: Float
                             , gsSpace     :: H.Space
                             , gsLevel     :: Mesh
                             , gsResources :: [FilePath]
                             , gsActives   :: [GameObject]
                             , gsInactives :: [GameObject]
                             }
    deriving Show
  
  data GameLevel = GameLevel { levLines  :: [LG.Line]
                             , levBounds :: LG.Rect
                             , levCarve  :: Double
                             , levMesh   :: Mesh
                             }
  
  generateLevel :: [LG.Line] -> LG.Rect -> Double -> GameLevel
  generateLevel ls b c = GameLevel{ levLines  = ls
                                  , levBounds = b
                                  , levCarve  = c
                                  , levMesh   = maskToMesh $
                                                rasterizeLines ls b c
                                  }
  
  level1 = generateLevel [((0, 8), (20, 12))]
                         ((0, 0), (20, 20))
                         2.5
  
  -------------------------------------------------------
  -- Game Functions
  -------------------------------------------------------
  
  -- | Return a GameState object set to reasonable initial values
  newGameState :: IO GameState
  newGameState = do
    space <- newSpaceWithGravity gravity
    cwd <- getCurrentDirectory
    
    let controllable obj = return (obj{ gBehavior = BeControllable })
    
    -- A function that tacks on an 'add' action for objects that don't do it for themselves
    let addRm obj = do
          let add = Action "add" $ sequence_ (map (space `includes`) (gPrim obj))
          -- let rm  = H.spaceRemove space obj
          return (obj { gAdd = add, gRemove = inaction }) -- TODO: add remove fn
    
    objs <- sequence [ createWorm space (H.Vector 0 1)    >>=   controllable    >>= addRm
                     , makeBox 0.3 (H.Vector 0 5)                               >>= addRm
                     , makeBox 0.8 (H.Vector (-3) 4)                            >>= addRm
                     , makeBox 0.7 (H.Vector (-2) 4.5)                          >>= addRm
                     , makeBox 0.6 (H.Vector (1.5) 6)                           >>= addRm
                     , makeBox 0.8 (H.Vector (2) 5)                             >>= addRm
                     , makeBox 0.4 (H.Vector (4) 4)                             >>= addRm
                     , makeWall (H.Vector (-3) (0)) (H.Vector (3) (0))          >>= addRm
                     , makeWall (H.Vector (-5) (2)) (H.Vector (0) (-2))         >>= addRm
                     , makeWall (H.Vector ( 5) (3)) (H.Vector (1) (-2.5))       >>= addRm
                     , makeWall (H.Vector (-1) (-2.5)) (H.Vector (1) (-2.5))    >>= addRm
                     ]
    
    -- Let each object add itself to the space
    forM_ objs $ (\ GameObject{ gAdd = add } -> act add )
    return (GameState (0.0) space (levMesh level1) [cwd] objs [])
  
  startGame :: IO ()
  startGame = do
    configureProjection (Perspective 60.0) Nothing
    stateRef <- newGameState >>= newIORef
    
    loadTexture "dirt-texture.png" 2
    
    -- mouseButtonCallback   $= processMouseInput stateRef
    windowSizeCallback $= resizeWindow
    
    now      <- get time
    gameLoop stateRef now
  
  -- | The main loop
  gameLoop :: IORef GameState -> Double -> IO ()
  gameLoop stateRef oldTime = do
    -- Some key states
    quitKey  <- getKey (SpecialKey ESC)
    
    acceptKeyboardCommands stateRef
    
    -- Update display and time
    updateDisplay stateRef
    newTime <- advanceTime stateRef oldTime
  
    -- Continue the game as long as quit signal has not been given
    when (quitKey /= Press)
      (gameLoop stateRef newTime)
  
  acceptKeyboardCommands :: IORef GameState -> IO ()
  acceptKeyboardCommands stateRef = do
    state <- get stateRef
    let controllables = ((gsActives state) `withBehavior` BeControllable)
    
    left  <- getKey (SpecialKey LEFT)
    right <- getKey (SpecialKey RIGHT)
    up    <- getKey (SpecialKey UP)
    down  <- getKey (SpecialKey DOWN)
    
    forM_ controllables $ \obj -> do
      let body  = H.getBody (getPrimShape $ head (gPrim obj))
      
      H.setForce body (H.Vector 0 0)
      
      when (left  == Press) (movePlayerLeft  body)
      when (right == Press) (movePlayerRight body)
      when (up    == Press) (movePlayerUp    body)
      when (down  == Press) (movePlayerDown  body)
    
    where movePlayerRight body = H.setForce body (H.Vector 1.5 0)
          movePlayerLeft  body = H.setForce body (H.Vector (-1.5) 0)
          movePlayerUp    body = H.setForce body (H.Vector 0 (1.5))
          movePlayerDown  body = H.setForce body (H.Vector 0 (-1.5))
    
  
  -- | Renders the current state.
  updateDisplay :: IORef GameState -> IO ()
  updateDisplay stateRef  = do
    state <- get stateRef
    clear [ColorBuffer, DepthBuffer]
    
    let controllables = ((gsActives state) `withBehavior` BeControllable)
        player = head controllables
    
    (H.Vector px_ py_) <- (H.getPosition $ H.getBody $ getPrimShape $ firstPrim player)
    let (px, py) = (realToFrac px_, realToFrac py_)
    (x, y) <- getMousePan
    
    
    lookingAt (Vector3 (px + x) (py + y) 2.5) -- Eye
              (Vector3 px py 0)               -- Target
              (Vector3 0 1 0) $ do
      moveLight $ Vector3 0.5 1.0 (-3.0)
      -- preservingMatrix $ do
      --   color $ Color3 0.6 0.3 (0.15 :: Float)
      --   scale 4 4 (-1.5 :: Float)
      --   translate $ Vector3 (-1) (-1) ( 5 :: Float)
      --   drawLevel (gsLevel state)
      forM_ (gsActives state) (act . gDraw)
    
    preservingMatrix $ do
      translate $ Vector3 (0) (0) (-10 :: Float)
      scale (14) (14) (0 :: Float)
      renderTexture 2 (-1) (-1) (2) (2)
    
    -- angle <- return (gsAngle state)
    -- stateRef $= state { gsAngle = angle + 0.01 }
    swapBuffers
  
  getMousePan :: IO (Double, Double)
  getMousePan = do
    Position cx cy <- get mousePos
    Size w h <- get windowSize
    let center p d = (fromIntegral (p - (d `div` 2))) / (fromIntegral d)
    if cx == 0 && cy == 0 then
      return ((center (w `div` 2) w) * (-1), center (h `div` 2) h)
      else return ((center cx w) * (-1), center cy h)
  
  innerBounds ary = ((a, b), (c - 1, d - 1))
    where ((a, b), (c, d)) = bounds ary
  
  maskToMesh :: DepthMask -> Mesh
  maskToMesh a = Mesh "tunnel" (0,0,0) vs ns fs
    where
      (w, h) = snd $ bounds a
      tup k = (k, k, k)
      rw = w + 1
      vs = map vertex (assocs a)
      ns = map normal (assocs a)
      fs = [[ tup ((i + 0) * rw + j + 1),
              tup ((i + 1) * rw + j + 1),
              tup ((i + 1) * rw + j + 2),
              tup ((i + 0) * rw + j + 2) ] |
            (i, j) <- range (innerBounds a) ]
      point i@(x, y)     = (fromIntegral x, fromIntegral y, (a ! i)/1)
      vertex ((x, y), d) = (fromIntegral x, fromIntegral y, -d)
      normal ((x, y), d) =
        let x_f = fromIntegral x
            y_f = fromIntegral y
            d1 = point (x + 1, y)
            d2 = point (x + 1, y + 1)
            d3 = (0, 0, 0)
            ok = inRange (bounds a) ((x + 1), (y + 1))
        in if ok then d1 `cross` d2 else d3
  
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
  
  createWorm :: H.Space -> H.Vector -> IO GameObject
  createWorm space pos@(H.Vector x y) = do
    -- Create worm segments
    let partPositions = [(H.Vector (x + i) y) | i <- [0,0.5..2.5]]
    prims <- mapM (makeCirclePrim 0.2 wormBody) partPositions
    
    -- Create pivots between body parts
    let pairs = zip (drop 0 prims) (drop 1 prims)
    joints <- mapM pivotJointBetween pairs
    
    fileData <- readFile "silkworm.obj"
    wormObj <- return $ readWaveFront fileData
    
    -- Draw the silkworm
    let draw = combineActions (map drawGamePrim prims)
    return $ gameObject{ gPrim = prims, gDraw = draw }

    -- let morph = mkMorph wormObj (fromIntegral $ length partPositions)
    --     -- ctrls = map (\(H.Vector x y) -> (realToFrac x, realToFrac y, 0)) partPositions
    --     draw = preservingMatrix $ do
    --              ctrls <- forM prims $ \p -> do
    --                let s = getPrimShape p
    --                (H.Vector x y) <- H.getPosition (H.getBody s)
    --                return ((realToFrac x), (realToFrac y), 0)
    --              let (x, y, z) = center ctrls
    --              translate $ Vector3 x y z
    --              color  $ Color3 0.2 0.7 (0.2 :: GLfloat)
    --              drawObject (blockMorph morph ctrls)
    
    -- return $ gameObject{ gPrim = prims, gDraw = Action "worm" draw }
    
    where pivotJointBetween (p1, p2) =
            let b1 = (H.getBody $ getPrimShape p1)
                b2 = (H.getBody $ getPrimShape p2)
            in do a <- H.getPosition b1
                  b <- H.getPosition b2
                  j <- H.newJoint b1 b2 (H.Pivot (midpoint a b))
                  H.spaceAdd space j
                  return j
  
  orientationForShape shape action = do
    (H.Vector x y) <- H.getPosition $ H.getBody shape
    angle          <- H.getAngle    $ H.getBody shape
    -- putStrLn $ "x: " ++ (show x) ++ ", y: " ++ (show y)
    preservingMatrix $ do
      translate    $ Vector3 x y 0
      rotate (toDegrees angle) $ Vector3 0 0 (1 :: GLfloat)
      action
    where toDegrees rad = rad / pi * 180
    
  ------------------------------------------------------------
  -- Simulation bookkeeping
  ------------------------------------------------------------

  -- | Advances the time.
  advanceTime :: IORef GameState -> Double -> IO Double
  advanceTime stateRef oldTime = do
    newTime <- get time
  
    -- Advance simulation
    let mult           = frameSteps / framePeriod
        framesPassed   = truncate $ mult * (newTime - oldTime)
        simulNewTime   = oldTime + toEnum framesPassed / mult
    advanceSimulTime stateRef $ min maxSteps framesPassed
  
    -- Correlate with reality
    newTime' <- get time
    let diff = newTime' - simulNewTime
        sleepTime = framePeriod - diff
    when (sleepTime > 0) $ sleep sleepTime
    return simulNewTime
    where
      -- | Advances the time in a certain number of steps.
      advanceSimulTime :: IORef GameState -> Int -> IO ()
      advanceSimulTime _        0     = return ()
      advanceSimulTime stateRef steps = do
        state <- get stateRef
        replicateM_ steps $ H.step (gsSpace state) frameDelta
      