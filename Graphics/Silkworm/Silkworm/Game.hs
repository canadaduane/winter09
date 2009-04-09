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
    mousePos,
    windowSizeCallback, windowSize,
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
  import Silkworm.OpenGLHelper (
    PerspectiveType(..),
    resizeWindow,
    configureProjection,
    lookingAt,
    moveLight )
  import Silkworm.LevelGenerator (rasterizeLines)
  import Silkworm.HipmunkHelper (newSpaceWithGravity, midpoint)
  import Silkworm.WaveFront (readWaveFront)
  import Silkworm.Object3D (Object3D(..), faces,
    Morph(..), mkMorph, blockMorph, crSplineN, center)
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
    gShape     :: H.Shape,
    gShapeType :: H.ShapeType,
    gSubstance :: Substance,
    gBehavior  :: Behavior,
    gMorph     :: Morph,
    gDraw      :: Action,
    gRemove    :: Action
  } deriving Show
  
  data Substance = Substance
                   H.Elasticity  -- Elasticity
                   H.Friction    -- Friction
                   Float         -- Density
    deriving Show
  
  rubber   = Substance 0.8 1.0 0.5
  wood     = Substance 0.3 0.8 0.5
  concrete = Substance 0.1 0.8 0.8
  
  act :: Action -> IO ()
  act (Action name fn) = do
    -- putStrLn name
    fn
  
  drawGameObject :: GameObject -> IO ()
  drawGameObject obj = act (gDraw obj)
  
  gameObject :: H.Shape -> GameObject
  gameObject shape =
    GameObject { gShape     = shape
               , gShapeType = H.Circle 1.0
               , gSubstance = wood
               , gBehavior  = BeStationary
               , gMorph     = NoMorph
               , gDraw      = inaction
               , gRemove    = inaction
               }
  
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
    ground <- createWall space (H.Vector (-1) (-0.5)) (H.Vector (1) (-0.5))
    square <- createBlock space (H.Vector 0 3) 0.4
    -- worm <- createWorm space
    -- sign <- createSign space
    -- level <- return $ maskToObject3D level1
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
  
  full = 1.0 :: Double
  zero = 0.0 :: Double
  white = Color3 full full full
  red = Color3 1 zero zero
  facingCamera = Normal3 0 0 (-1) :: Normal3 Double
  
  drawShape :: H.Shape -> H.ShapeType -> IO ()
  -- Circles
  drawShape shape (H.Circle radius) = do
    H.Vector px py <- H.getPosition $ H.getBody shape
    angle          <- H.getAngle    $ H.getBody shape
    normal facingCamera
    color white
    renderPrimitive LineStrip $ do
      let segs = 20; coef = 2*pi/toEnum segs
      forM_ [0..segs] $ \i -> do
        let r = toEnum i * coef
            x = radius * cos (r + angle) + px
            y = radius * sin (r + angle) + py
        vertex (Vertex3 (realToFrac x) (realToFrac y) zero)
      vertex (Vertex3 (realToFrac px) (realToFrac py) zero)
    drawPoint (H.Vector px py)
  -- Polygons
  drawShape shape (H.Polygon verts) = do
    pos   <- H.getPosition $ H.getBody shape
    angle <- H.getAngle    $ H.getBody shape
    let rot = H.rotate $ H.fromAngle angle
        verts' = map ((+pos) . rot) verts
    normal facingCamera
    color white
    renderPrimitive LineStrip $ do
      forM_ (verts' ++ [head verts']) $ \(H.Vector x y) -> do
        vertex (Vertex3 (realToFrac x) (realToFrac y) zero)
    drawPoint pos
  -- Line Segments
  drawShape shape (H.LineSegment p1 p2 _) = do
    let v (H.Vector x y) = vertex (Vertex3 (realToFrac x) (realToFrac y) zero)
    pos <- H.getPosition $ H.getBody shape
    normal facingCamera
    color white
    renderPrimitive Lines $ v (p1 + pos) >> v (p2 + pos)
    drawPoint pos

  -- | Draws a red point.
  drawPoint :: H.Vector -> IO ()
  drawPoint (H.Vector px py) = do
    color red
    renderPrimitive Points $ do
      vertex (Vertex3 (realToFrac px) (realToFrac py) zero)

  -- | Shortcut function for making rectangle shapes for the physics engine
  rect :: Float -> Float -> H.ShapeType
  rect w h = H.Polygon verts
    where verts = map (uncurry H.Vector)
                      [(-w/2,-h/2)
                      ,(-w/2, h/2)
                      ,( w/2, h/2)
                      ,( w/2,-h/2)]
  
  -- become :: GameObject -> IO ()
  -- become (GameObject { gShape = shape, gShapeType = stype, gSubstance = subst }) = do
  --     H.setElasticity shape e
  --     H.setFriction shape f
  --     return ()
  --   where (Substance e f d) = subst
  --         mass = 
  
  -- | Create a new (body, shape) pair with the specified characteristics
  newShape :: H.ShapeType -> Float -> Float -> Float -> IO H.Shape
  newShape stype mass elast fric = do
    body  <- H.newBody mass moment
    shape <- H.newShape body stype (H.Vector 0 0)
    H.setElasticity shape elast
    H.setFriction shape fric
    return shape
    where moment = case stype of
            H.Circle radius           -> H.momentForCircle mass (0, radius) (H.Vector 0 0)
            H.Polygon verts           -> H.momentForPoly mass verts (H.Vector 0 0)
            H.LineSegment p1 p2 thick -> H.infinity
  
  newCircle :: Float -> Float -> IO (H.Shape, H.ShapeType)
  newCircle mass radius = do
      shape <- newShape stype mass 0.3 0.0
      return (shape, stype)
    where stype = (H.Circle radius)
  
  newSquare :: Float -> Float -> IO (H.Shape, H.ShapeType)
  newSquare mass side   = do
      shape <- newShape stype mass 0.3 0.2
      return (shape, stype)
    where stype = (rect side side)
  
  newLine :: H.Vector -> H.Vector -> Float -> IO (H.Shape, H.ShapeType)
  newLine p1 p2 thick   = do
      shape <- newShape stype H.infinity 0.6 1.0
      return (shape, stype)
    where stype = (H.LineSegment p1 p2 thick)
  
  -- | Include a (body, shape) pair in the physics space's calculations
  includeShape :: H.Space -> H.Shape -> IO ()
  includeShape space shape = do
    H.spaceAdd space (H.getBody shape)
    H.spaceAdd space shape
    return ()
  
  createBlock :: H.Space -> H.Vector -> Float -> IO GameObject
  createBlock space position size = do
    (shape, stype) <- newSquare 3.0 size
    includeShape space shape
    H.setPosition (H.getBody shape) position
    let draw = drawShape shape stype
    return $ (gameObject shape) { gDraw = Action "draw wall" draw }
  
  createWall :: H.Space -> H.Vector -> H.Vector -> IO GameObject
  createWall space p1 p2 = do
    (shape, stype) <- newLine p1 p2 0.05
    -- includeShape space shape
    H.spaceAdd space shape -- Only add shape so that walls are static
    H.setPosition (H.getBody shape) (midpoint p1 p2)
    let draw = drawShape shape stype
    return $ (gameObject shape) { gDraw = Action "draw wall" draw }
  
  wormSegments = 10 :: Integer
  
  createWorm :: H.Space -> IO GameObject
  createWorm space = do
    fileData <- readFile "silkworm.obj"
    wormObj <- return $ readWaveFront fileData
    
    -- let ctrls = crSplineN wormSegments [(-2,0,0),(-1,0,0),(0,0.5,0),(1,0,0),(2,0,0)]
    let morph@Morph{ mphControls = m_ctrls } = mkMorph wormObj wormSegments
        segs = zip [0..] m_ctrls
    
    -- Generate the worm body parts
    shapes <- forM segs $ \(i, (x, y, z)) -> do
      shape <- newShape (H.Circle 0.2) 2.0 0.2 0.0
      includeShape space shape
      H.setPosition (H.getBody shape) (H.Vector ((realToFrac x) + 1.5) ((realToFrac y) + 2.5))
      H.setVelocity (H.getBody shape) (H.Vector 0 ((fromIntegral i)/10))
      return shape
    
    -- Create pivots between body parts
    let pairs = zip (drop 0 shapes) (drop 1 shapes)
    forM_ pairs $ \(s1, s2) -> do
      p1 <- H.getPosition (H.getBody s1)
      p2 <- H.getPosition (H.getBody s2)
      j <- H.newJoint (H.getBody s1) (H.getBody s2) (H.Pivot (midpoint p1 p2))
      H.spaceAdd space j
    
    let draw = do
          -- putStrLn "draw silkworm"
          ctrls <- forM shapes $ \s -> do
            (H.Vector x y) <- H.getPosition (H.getBody s)
            return ((realToFrac x), (realToFrac y), 0)
          
          preservingMatrix $ do
            let (x, y, z) = center ctrls
            translate $ Vector3 x (y-0) 0
            color  $ Color3 0.2 0.7 (0.2 :: GLfloat)
            drawObject (blockMorph morph ctrls)
    
    mainShape <- return (head shapes)
    return $ (gameObject mainShape) { gDraw = Action "draw silkworm" draw
                                      , gMorph = morph }
  
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
      