module Silkworm.GameObject
  ( GameShape(..)
  , GameObject(..)
  , gameObject
  , makeObjectWithMass
  , makeObject
  , makeCircleObject
  , makeRectangleObject
  , makeSquareObject
  , makeLineObject
  , makeStaticLineObject
  , makeWall
  , makeBox
  , drawGameObject
  , getHipShape
  , getHipShapeType
  , includes
  ) where
  
  import Control.Monad (forM, forM_, when)
  
  import Graphics.UI.GLFW
  import Graphics.Rendering.OpenGL as GL
  
  import Physics.Hipmunk as H
  
  import Silkworm.Action
  import Silkworm.Behavior
  import Silkworm.HipmunkHelper
  import Silkworm.Mesh
  import Silkworm.Substance
  
  
  data GameShape = GameShape Shape ShapeType
                 | NoGameShape
    deriving Show
  
  -- The Game Object : Here is where we define everything to do with each moving thing
  data GameObject = GameObject {
    gShape     :: GameShape,
    gSubstance :: Substance,
    gBehavior  :: Behavior,
    gMorph     :: Morph,
    gDraw      :: Action
  } deriving Show
  
  gameObject :: GameObject
  gameObject = GameObject NoGameShape genericSubstance BeStationary NoMorph inaction
  
  makeObjectWithMass :: Float -> ShapeType -> Substance -> Vector -> IO GameObject
  makeObjectWithMass mass stype subst pos = do
      body  <- newBody mass moment
      shape <- newShape body stype (Vector 0 0)
      setElasticity shape elasticity
      setFriction shape friction
      setPosition body pos
      let gameShape = GameShape shape stype
      return gameObject { gShape     = gameShape
                        , gSubstance = subst
                        , gDraw      = action (drawShapeTrace gameShape)
                        }
    where (Substance elasticity friction _) = subst
          moment = case stype of
            H.Circle radius           -> momentForCircle mass (0, radius) (Vector 0 0)
            H.Polygon verts           -> momentForPoly mass verts (Vector 0 0)
            H.LineSegment p1 p2 thick -> infinity
  
  makeObject :: ShapeType -> Substance -> Vector -> IO GameObject
  makeObject stype subst pos = makeObjectWithMass mass stype subst pos
    where (Substance _ _ density) = subst
          mass = density * (area stype)
  
  makeCircleObject :: Float -> Substance -> Vector -> IO GameObject
  makeCircleObject radius = makeObject circle
    where circle = Circle radius
  
  makeRectangleObject :: Float -> Float -> Substance -> Vector -> IO GameObject
  makeRectangleObject w h = makeObject (rectangle w h)
  
  makeSquareObject :: Float -> Substance -> Vector -> IO GameObject
  makeSquareObject s = makeRectangleObject s s
  
  makeLineObject :: H.Position -> H.Position -> Substance -> IO GameObject
  makeLineObject p1 p2 s = makeObject (LineSegment p1 p2 lineThickness) s (midpoint p1 p2)
  
  makeStaticLineObject :: H.Position -> H.Position -> Substance -> IO GameObject
  makeStaticLineObject p1 p2 s = makeObjectWithMass infinity (LineSegment p1 p2 lineThickness) s (midpoint p1 p2)
  
  makeWall :: Vector -> Vector -> IO GameObject
  makeWall p1 p2 = makeStaticLineObject p1 p2 wood
  
  makeBox :: Float -> Vector -> IO GameObject
  makeBox size pos = makeSquareObject size wood pos
  
  drawGameObject :: GameObject -> IO ()
  drawGameObject obj = act (gDraw obj)
  
  getHipShape :: GameObject -> Shape
  getHipShape (GameObject { gShape = gs }) = shape
    where (GameShape shape stype) = gs
  
  getHipShapeType :: GameObject -> ShapeType
  getHipShapeType (GameObject { gShape = gs }) = stype
    where (GameShape shape stype) = gs
  
  includes :: Space -> GameObject -> IO ()
  includes space g = do
      mass <- getMass body
      -- Only add bodies of non-infinitely massive objects
      when (mass /= infinity) (spaceAdd space body)
      spaceAdd space shape
    where shape = getHipShape g
          body = getBody shape
  
  
  
  -- Functions local to this module
  
  lineThickness = 0.05
  
  -- | Shortcut function for making rectangle shapes for the physics engine
  rectangle :: Float -> Float -> ShapeType
  rectangle w h = H.Polygon verts
    where verts = map (uncurry Vector)
                      [ (-w/2,-h/2)
                      , (-w/2, h/2)
                      , ( w/2, h/2)
                      , ( w/2,-h/2)
                      ]
  
  full = 1.0 :: Double
  zero = 0.0 :: Double
  white = Color3 full full full
  red = Color3 1 zero zero
  facingCamera = Normal3 0 0 (-1) :: Normal3 Double
  
  drawShapeTrace :: GameShape -> IO ()
  drawShapeTrace (GameShape shape stype) = do
      -- Get center and angle of object
      pos@(Vector px py) <- getPosition body
      angle              <- getAngle    body
      -- Set OpenGL to white lines
      traceInWhite
      case stype of
        H.Circle radius           -> circle radius angle px py
        H.Polygon verts           -> polygon verts angle pos
        H.LineSegment p1 p2 thick -> line p1 p2 pos
      -- Finish with a dot at the center
      drawPoint pos
    where body  = getBody shape
          circle radius angle px py =
            renderPrimitive LineStrip $ do
              let segs = 20; coef = 2*pi/toEnum segs
              forM_ [0..segs] $ \i -> do
                let r = toEnum i * coef
                    x = radius * cos (r + angle) + px
                    y = radius * sin (r + angle) + py
                vertex (Vertex3 (realToFrac x) (realToFrac y) zero)
              vertex (Vertex3 (realToFrac px) (realToFrac py) zero)
          polygon verts angle pos =
            let rot = H.rotate $ fromAngle angle
                verts' = map ((+pos) . rot) verts
            in renderPrimitive LineStrip $ do
                 forM_ (verts' ++ [head verts']) $ \(Vector x y) -> do
                   vertex (Vertex3 (realToFrac x) (realToFrac y) zero)
          line p1 p2 pos =
            let v (H.Vector x y) = vertex (Vertex2 x y)
            in renderPrimitive Lines $ v (p1 + pos) >> v (p2 + pos)
  
  traceInWhite :: IO ()
  traceInWhite = do
    normal facingCamera
    color white
  
  -- | Draws a red point.
  drawPoint :: Vector -> IO ()
  drawPoint (Vector px py) = do
    color red
    renderPrimitive Points $ do
      vertex (Vertex3 (realToFrac px) (realToFrac py) zero)
