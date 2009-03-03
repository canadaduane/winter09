module Main (main) where

import Control.Monad
import Data.IORef
import qualified Data.Map as M
import System.Exit

import Graphics.UI.GLFW
import Graphics.Rendering.OpenGL
import qualified Physics.Hipmunk as H


------------------------------------------------------------
-- Some constants
------------------------------------------------------------

-- | Desired (and maximum) frames per second.
desiredFPS :: Int
desiredFPS = 60

-- | How much seconds a frame lasts.
framePeriod :: Double
framePeriod = 1 / toEnum desiredFPS

-- | How many steps should be done per frame.
frameSteps :: Double
frameSteps = 6

-- | Maximum number of steps per frame (e.g. if lots of frames get
--   dropped because the window was minimized)
maxSteps :: Int
maxSteps = 20

-- | How much time should pass in each step.
frameDelta :: H.Time
frameDelta = 3.33e-3

-- | How much slower should the slow mode be.
slowdown :: Double
slowdown = 10

-- | 0 :: GLfloat
zero :: GLfloat
zero = 0

-- | Asserts that an @IO@ action returns @True@, otherwise
--   fails with the given message.
assertTrue :: IO Bool -> String -> IO ()
assertTrue act msg = do {b <- act; when (not b) (fail msg)}




------------------------------------------------------------
-- State
------------------------------------------------------------

-- | Our current program state that will be passed around.
data State = State {
      stSpace  :: H.Space,
      stShapes :: M.Map H.Shape (IO () {- Drawing -}
                                ,IO () {- Removal -})
    }

-- | Our initial state.
initialState :: IO State
initialState = do
  -- The (empty) space
  space  <- H.newSpace
  H.setElasticIterations space 10
  H.setGravity space $ H.Vector 0 (-230)

  -- The ground
  static <- H.newBody H.infinity H.infinity
  H.setPosition static (H.Vector (-330) 0)
  let seg1type = H.LineSegment (H.Vector 50  (-230))
                               (H.Vector 610 (-230)) 1
  seg1   <- H.newShape static seg1type 0
  H.setFriction seg1 1.0
  H.setElasticity seg1 0.6
  H.spaceAdd space (H.Static seg1)

  -- The seesaw
  ---- Support
  let supportV = map (uncurry H.Vector) [(-15,-20),(-5,20),(5,20),(15,-20)]
      supportT = H.Polygon supportV
      supportM = 500
      supportI = H.momentForPoly supportM supportV 0
  supportB <- H.newBody supportM supportI
  H.setPosition supportB (H.Vector 0 (20-230))
  supportS <- H.newShape supportB supportT 0
  H.setFriction supportS 2.0
  H.setElasticity supportS 0.1
  H.spaceAdd space supportB
  H.spaceAdd space supportS
  ----- Board
  let boardV = map (uncurry H.Vector) [(-100,1),(100,1),(100,-1),(-100,-1)]
      boardT = H.Polygon boardV
      boardM = 10
      boardI = H.momentForPoly boardM boardV 0
  boardB <- H.newBody boardM boardI
  H.setPosition boardB (H.Vector 0 (40-230))
  boardS <- H.newShape boardB boardT 0
  let setBoardProps shape = do
        H.setFriction shape 2.0
        H.setElasticity shape 0.1
  setBoardProps boardS
  H.spaceAdd space boardB
  H.spaceAdd space boardS
  boardS2 <- forM (zip boardV $ tail $ cycle boardV) $ \(v1,v2) -> do
    seg <- H.newShape boardB (H.LineSegment v1 v2 0.1) 0
    setBoardProps seg
    H.spaceAdd space seg
    return seg
  ----- Joint
  seesawJoint <- H.newJoint supportB boardB (H.Pin (H.Vector 0 20) 0)
  H.spaceAdd space seesawJoint
  ----- Avoiding self-collisions
  forM_ (supportS : boardS : boardS2) $ \s -> do
    H.setGroup s 1
  ----- Removing and drawing
  let drawSeeSaw = do
        drawMyShape supportS supportT
        drawMyShape boardS boardT
  let removeSeeSaw = do
        H.spaceRemove space supportB
        H.spaceRemove space supportS
        H.spaceRemove space boardB
        H.spaceRemove space boardS
        forM_ boardS2 (H.spaceRemove space)
        H.spaceRemove space seesawJoint


  return $ State space $ M.fromList
    [(seg1, (drawMyShape seg1 seg1type, return ()))
    ,(supportS, (drawSeeSaw, removeSeeSaw))]

-- | Destroy a state.
destroyState :: State -> IO ()
destroyState (State {stSpace = space}) = do
  H.freeSpace space




------------------------------------------------------------
-- Main function and main loop
------------------------------------------------------------

-- | Entry point.
main :: IO ()
main = do
  -- Initialize Chipmunk, GLFW and our state
  H.initChipmunk
  assertTrue initialize "Failed to init GLFW"
  stateVar <- initialState >>= newIORef

  -- Create a window
  assertTrue (openWindow (Size 800 600) [] Window) "Failed to open a window"
  windowTitle $= "Hipmunk Playground"

  -- Define some GL parameters for the whole program
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

  -- Add some callbacks
  windowCloseCallback   $= exitWith ExitSuccess
  mouseButtonCallback   $= processMouseInput stateVar

  -- Let's go!
  now <- get time
  loop stateVar now

-- | The simulation loop.
loop :: IORef State -> Double -> IO ()
loop stateVar oldTime = do
  -- Some key states
  slowKey  <- getKey (SpecialKey ENTER)
  quitKey  <- getKey (SpecialKey ESC)
  clearKey <- getKey (SpecialKey DEL)

  -- Quit?
  when (quitKey == Press) (terminate >> exitWith ExitSuccess)

  -- Clear?
  when (clearKey == Press) $ do
    destroyState =<< readIORef stateVar
    initialState >>= writeIORef stateVar

  -- Update display and time
  updateDisplay stateVar slowKey
  newTime <- advanceTime stateVar oldTime slowKey
  loop stateVar newTime

-- | Advances the time.
advanceTime :: IORef State -> Double -> KeyButtonState -> IO Double
advanceTime stateVar oldTime slowKey = do
  newTime <- get time

  -- Advance simulation
  let slower = if slowKey == Press then slowdown else 1
      mult   = frameSteps / (framePeriod * slower)
      framesPassed   = truncate $ mult * (newTime - oldTime)
      simulNewTime   = oldTime + toEnum framesPassed / mult
  advanceSimulTime stateVar $ min maxSteps framesPassed

  -- Correlate with reality
  newTime' <- get time
  let diff = newTime' - simulNewTime
      sleepTime = ((framePeriod * slower) - diff) / slower
  when (sleepTime > 0) $ sleep sleepTime
  return simulNewTime






------------------------------------------------------------
-- Display related functions
------------------------------------------------------------


-- | Renders the current state.
updateDisplay :: IORef State -> KeyButtonState -> IO ()
updateDisplay stateVar slowKey  = do
  state <- get stateVar
  clear [ColorBuffer]
  drawInstructions
  when (slowKey == Press) drawSlowMotion
  forM_ (M.assocs $ stShapes state) (fst . snd) -- Draw each one
  swapBuffers

drawInstructions :: IO ()
drawInstructions = preservingMatrix $ do
  translate (Vector3 (-320) 240 zero)
  scale 0.75 0.75 (zero + 1)
  let render str = do
        translate (Vector3 zero (-16) zero)
        renderString Fixed8x16 str

  color $ Color3 zero zero 1
  render "Press the left mouse button to create a ball."
  render "Press the right mouse button to create a square."
  render "Press the middle mouse button to create a triangle on a pendulum."

  color $ Color3 1 zero zero
  render "Hold LEFT SHIFT to create counterclockwise rotating objects."
  render "Hold RIGHT SHIFT to create clockwise rotating objects."
  render "Hold ENTER to see in slow motion."

  color $ Color3 zero zero zero
  render "Press DEL to clear the screen."

drawSlowMotion :: IO ()
drawSlowMotion = preservingMatrix $ do
  scale 2 2 (zero + 1)
  translate (Vector3 (-40) zero zero)
  color $ Color3 zero 1 zero
  renderString Fixed8x16 "Slowwww..."

-- | Draws a shape (assuming zero offset)
drawMyShape :: H.Shape -> H.ShapeType -> IO ()
drawMyShape shape (H.Circle radius) = do
  H.Vector px py <- H.getPosition $ H.getBody shape
  angle          <- H.getAngle    $ H.getBody shape

  color $ Color3 zero zero zero
  renderPrimitive LineStrip $ do
    let segs = 20; coef = 2*pi/toEnum segs
    forM_ [0..segs] $ \i -> do
      let r = toEnum i * coef
          x = radius * cos (r + angle) + px
          y = radius * sin (r + angle) + py
      vertex (Vertex2 x y)
    vertex (Vertex2 px py)
  drawPoint (H.Vector px py)
drawMyShape shape (H.LineSegment p1 p2 _) = do
  let v (H.Vector x y) = vertex (Vertex2 x y)
  pos <- H.getPosition $ H.getBody shape
  color $ Color3 zero zero zero
  renderPrimitive Lines $ v (p1 + pos) >> v (p2 + pos)
  drawPoint pos
drawMyShape shape (H.Polygon verts) = do
  pos   <- H.getPosition $ H.getBody shape
  angle <- H.getAngle    $ H.getBody shape
  let rot = H.rotate $ H.fromAngle angle
      verts' = map ((+pos) . rot) verts
  color $ Color3 zero zero zero
  renderPrimitive LineStrip $ do
    forM_ (verts' ++ [head verts']) $ \(H.Vector x y) -> do
      vertex (Vertex2 x y)
  drawPoint pos

-- | Draws a red point.
drawPoint :: H.Vector -> IO ()
drawPoint (H.Vector px py) = do
  color $ Color3 1 zero zero
  renderPrimitive Points $ do
    vertex (Vertex2 px py)






------------------------------------------------------------
-- Input processing
------------------------------------------------------------

-- | Returns the current mouse position in our space's coordinates.
getMousePos :: IO H.Position
getMousePos = do
  Position cx cy <- get mousePos
  Size _ h <- get $ windowSize
  model    <- get $ matrix (Just $ Modelview 0)
  proj     <- get $ matrix (Just Projection)
  view     <- get $ viewport
  let src = Vertex3 (fromIntegral cx) (fromIntegral $ h - cy) 0
  Vertex3 mx my _ <- unProject src (model :: GLmatrix GLdouble) proj view
  return $ H.Vector (realToFrac mx) (realToFrac my)

-- | Process a user mouse button press.
processMouseInput :: IORef State -> MouseButton -> KeyButtonState -> IO ()
processMouseInput _        _   Press   = return ()
processMouseInput stateVar btn Release = do
  rotateKeyCCW <- getKey (SpecialKey LSHIFT)
  rotateKeyCW  <- getKey (SpecialKey RSHIFT)
  let angVel = case (rotateKeyCCW, rotateKeyCW) of
                 (Press,   Release) -> 50
                 (Release, Press)   -> (-50)
                 _                  -> 0
  (shape,add,draw,remove) <- (case btn of
    ButtonLeft  -> createCircle
    ButtonRight -> createSquare
    _           -> createTriPendulum) angVel

  state <- get stateVar
  let space = stSpace state
  add space >> stateVar $= state {
    stShapes = M.insert shape (draw, remove space) $ stShapes state}





------------------------------------------------------------
-- Object creation
------------------------------------------------------------


-- | The return of functions that create objects.
type Creation = (H.Shape,          -- ^ A representative shape
                 H.Space -> IO (), -- ^ Function that add the entities
                 IO (),            -- ^ Function that draws the entity
                 H.Space -> IO ()  -- ^ Function that removes the entities
                )

-- | The type of the functions that create objects.
type Creator = H.CpFloat -> IO Creation

createCircle :: Creator
createCircle angVel = do
  let mass   = 20
      radius = 20
      t = H.Circle radius
  b <- H.newBody mass $ H.momentForCircle mass (0, radius) 0
  s <- H.newShape b t 0
  H.setAngVel b angVel
  H.setPosition b =<< getMousePos
  H.setFriction s 0.5
  H.setElasticity s 0.9
  let add space = do
        H.spaceAdd space b
        H.spaceAdd space s
  let draw = do
        drawMyShape s t
  let remove space = do
        H.spaceRemove space b
        H.spaceRemove space s
  return (s,add,draw,remove)

createSquare :: Creator
createSquare angVel = do
  let mass  = 18
      verts = map (uncurry H.Vector)
              [(-15,-15), (-15,15), (15,15), (15,-15)]
      t = H.Polygon verts
  b <- H.newBody mass $ H.momentForPoly mass verts 0
  s <- H.newShape b t 0
  H.setAngVel b angVel
  H.setPosition b =<< getMousePos
  H.setFriction s 0.5
  H.setElasticity s 0.6
  let add space = do
        H.spaceAdd space b
        H.spaceAdd space s
  let draw = do
        drawMyShape s t
  let remove space = do
        H.spaceRemove space b
        H.spaceRemove space s
  return (s,add,draw,remove)

createTriPendulum :: Creator
createTriPendulum angVel = do
  let mass  = 100
      verts = map (uncurry H.Vector) [(-30,-30), (0, 37), (30, -30)]
      t = H.Polygon verts
  b <- H.newBody mass $ H.momentForPoly mass verts 0
  s <- H.newShape b t 0
  H.setAngVel b angVel
  H.setPosition b =<< getMousePos
  H.setFriction s 0.8
  H.setElasticity s 0.3

  let staticPos = H.Vector 0 240
  static <- H.newBody H.infinity H.infinity
  H.setPosition static staticPos
  j <- H.newJoint static b (H.Pin 0 0)

  let add space = do
        H.spaceAdd space b
        H.spaceAdd space s
        H.spaceAdd space j
  let remove space = do
        H.spaceRemove space b
        H.spaceRemove space s
        H.spaceRemove space j
  let draw = do
        H.Vector x1 y1 <- H.getPosition b
        let H.Vector x2 y2 = staticPos
        color $ Color3 (zero+0.7) 0.7 0.7
        renderPrimitive LineStrip $ do
          vertex (Vertex2 x1 y1)
          vertex (Vertex2 x2 y2)
        drawMyShape s t
  return (s,add,draw,remove)






------------------------------------------------------------
-- Simulation bookkeeping
------------------------------------------------------------

-- | Advances the time in a certain number of steps.
advanceSimulTime :: IORef State -> Int -> IO ()
advanceSimulTime _        0     = return ()
advanceSimulTime stateVar steps = do
  removeOutOfSight stateVar
  state <- get stateVar
  replicateM_ steps $ H.step (stSpace state) frameDelta

-- | Removes all shapes that may be out of sight forever.
removeOutOfSight :: IORef State -> IO ()
removeOutOfSight stateVar = do
  state   <- get stateVar
  shapes' <- foldM f (stShapes state) $ M.assocs (stShapes state)
  stateVar $= state {stShapes = shapes'}
    where
      f shapes (shape, (_,remove)) = do
        H.Vector x y <- H.getPosition $ H.getBody shape
        if y < (-350) || abs x > 800
          then remove >> return (M.delete shape shapes)
          else return shapes
