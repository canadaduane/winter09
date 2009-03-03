module Silkworm.Drawing where
  
  import Control.Monad (forM_)
  import Graphics.Rendering.OpenGL as GL
  import Physics.Hipmunk as H
  import Silkworm.Constants

  -- | 0 :: GLfloat
  zero :: GLfloat
  zero = 0
  
  -- | Draws a shape (assuming zero offset)
  drawShape :: Shape -> ShapeType -> IO ()
  drawShape shape (H.Circle radius) = do
    Vector px py <- getPosition shapeBody
    angle        <- getAngle    shapeBody

    color $ Color3 zero zero zero
    renderPrimitive LineStrip $ do
      let segs = 20; coef = 2*pi/toEnum segs
      forM_ [0..segs] $ \i -> do
        let r = toEnum i * coef
            x = radius * cos (r + angle) + px
            y = radius * sin (r + angle) + py
        vertex (Vertex2 x y)
      vertex (Vertex2 px py)
    drawPoint (Vector px py)
    where
      shapeBody = getBody shape
  
  drawShape shape (H.LineSegment p1 p2 _) = do
    let v (Vector x y) = vertex (Vertex2 x y)
    pos <- getPosition shapeBody
    color $ Color3 zero zero zero
    renderPrimitive Lines $ v (p1 + pos) >> v (p2 + pos)
    drawPoint pos
    where
      shapeBody = getBody shape
  
  drawShape shape (H.Polygon verts) = do
    pos   <- getPosition shapeBody
    angle <- getAngle    shapeBody
    let rot    = H.rotate (fromAngle angle)
        verts' = map ((+pos) . rot) verts
    color $ Color3 zero zero zero
    renderPrimitive LineStrip $ do
      forM_ (verts' ++ [head verts']) $ \(Vector x y) -> do
        vertex (Vertex2 x y)
    drawPoint pos
    where
      shapeBody = getBody shape
  
  -- | Draws a red point.
  drawPoint :: Vector -> IO ()
  drawPoint (Vector px py) = do
    color $ Color3 1 zero zero
    renderPrimitive Points $ do
      vertex (Vertex2 px py)