module Silkworm.GameObject
  ( GamePrim(..)
  , GameObject(..)
  , gameObject
  , makePrimWithMass
  , makePrim
  , makeCirclePrim
  , makeRectanglePrim
  , makeSquarePrim
  , makeLinePrim
  , makeStaticLinePrim
  , makeWall
  , makeBox
  -- , drawGameObject
  , withBehavior
  , getPrimShape
  , getPrimShapeType
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
  
  -- A Game Primitive holds shape and substance info for pieces of objects
  --   e.g. silkworm is composed of several primitives, specifically Circle objects
  data GamePrim = GamePrim Shape ShapeType Substance
    deriving Show
  
  -- The Game Object : Here is where we define everything to do with each moving thing
  data GameObject = GameObject {
    gPrim      :: [GamePrim],
    gBehavior  :: Behavior,
    gMorph     :: Morph,
    gDraw      :: Action,
    gAdd       :: Action,
    gRemove    :: Action
  } deriving Show
  
  gameObject :: GameObject
  gameObject = GameObject [] BeStationary NoMorph inaction inaction inaction
  
  makePrimWithMass :: Float -> ShapeType -> Substance -> Vector -> IO GamePrim
  makePrimWithMass mass stype subst pos = do
      body  <- newBody mass moment
      shape <- newShape body stype (Vector 0 0)
      setElasticity shape elasticity
      setFriction shape friction
      setPosition body pos
      return $ GamePrim shape stype subst
    where (Substance elasticity friction _) = subst
          moment = case stype of
            H.Circle radius           -> momentForCircle mass (0, radius) (Vector 0 0)
            H.Polygon verts           -> momentForPoly mass verts (Vector 0 0)
            H.LineSegment p1 p2 thick -> infinity
  
  makePrim :: ShapeType -> Substance -> Vector -> IO GamePrim
  makePrim stype subst pos = makePrimWithMass mass stype subst pos
    where (Substance _ _ density) = subst
          mass = density * (area stype)
  
  makeCirclePrim :: Float -> Substance -> Vector -> IO GamePrim
  makeCirclePrim radius = makePrim circle
    where circle = Circle radius
  
  makeRectanglePrim :: Float -> Float -> Substance -> Vector -> IO GamePrim
  makeRectanglePrim w h = makePrim (rectangle w h)
  
  makeSquarePrim :: Float -> Substance -> Vector -> IO GamePrim
  makeSquarePrim s = makeRectanglePrim s s
  
  makeLinePrim :: H.Position -> H.Position -> Substance -> IO GamePrim
  makeLinePrim p1 p2 s = makePrim (LineSegment p1 p2 lineThickness) s (midpoint p1 p2)
  
  makeStaticLinePrim :: H.Position -> H.Position -> Substance -> IO GamePrim
  makeStaticLinePrim p1 p2 s = makePrimWithMass infinity (LineSegment p1 p2 lineThickness) s (midpoint p1 p2)
  
  makeWall :: Vector -> Vector -> IO GamePrim
  makeWall p1 p2 = makeStaticLinePrim p1 p2 wood
  
  makeBox :: Float -> Vector -> IO GamePrim
  makeBox size pos = makeSquarePrim size wood pos
  
  -- drawGamePrim :: GamePrim -> IO ()
  -- drawGamePrim (GamePrim ) = act (gDraw obj)
  
  withBehavior :: [GameObject] -> Behavior -> [GameObject]
  withBehavior gs b = filter behaving gs
    where behaving g = (gBehavior g) == b
  
  getPrimShape :: GamePrim -> Shape
  getPrimShape (GamePrim shape _ _) = shape
  
  getPrimShapeType :: GamePrim -> ShapeType
  getPrimShapeType (GamePrim _ stype _) = stype
  
  includes :: Space -> GamePrim -> IO ()
  includes space prim = do
      mass <- getMass body
      -- Only add bodies of non-infinitely massive objects
      when (mass /= infinity) (spaceAdd space body)
      spaceAdd space shape
    where shape = getPrimShape prim
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
  
  drawShapeTrace :: GamePrim -> IO ()
  drawShapeTrace (GamePrim shape stype _) = do
      -- Get center and angle of object
      pos@(Vector px py) <- getPosition body
      angle              <- getAngle    body
      -- Set OpenGL to white lines
      
      preservingMatrix $ do
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
