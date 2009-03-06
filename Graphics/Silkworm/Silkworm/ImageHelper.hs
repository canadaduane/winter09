module Silkworm.ImageHelper (loadTexture, renderTexture) where

  import System.Directory (getCurrentDirectory)
  import System.Exit (ExitCode(..), exitWith)
  import Codec.Image.PNG (loadPNGFile, imageData, dimensions, hasAlphaChannel)
  import Data.Array.Storable (withStorableArray)
  import Graphics.Rendering.OpenGL
  
  -- | Loads a PNG image file into an OpenGL texture buffer with id textureId
  loadTexture :: FilePath -> GLuint -> IO ()
  loadTexture filename textureId = do
    cwd <- getCurrentDirectory
    loaded <- loadPNGFile (cwd ++ "/" ++ filename)
    case loaded of
      Left  msg -> (putStrLn $ "Error: " ++ msg) >> exitWith (ExitFailure (-1))
      Right img -> do
             texture        Texture2D $= Enabled
             textureBinding Texture2D $= Just (TextureObject textureId)
             let (w,h) = dimensions img
                 (a,b) = case hasAlphaChannel img of
                           True  -> (RGBA8, RGBA)
                           False -> (RGB8,  RGB)
             withStorableArray (imageData img) $ \ptr ->
                 build2DMipmaps Texture2D a (fromIntegral w) (fromIntegral h) (PixelData b UnsignedByte ptr)
  
  -- | Show a texture on a rectangular space
  renderTexture :: GLuint -> GLfloat -> GLfloat -> GLfloat -> GLfloat -> IO ()
  renderTexture tId x y w h = do
    texture Texture2D $= Enabled
    textureBinding Texture2D $= Just (TextureObject tId)
    color $ Color3 1.0 1.0 (1.0 :: GLfloat)
    renderPrimitive Quads $ do texCoord (TexCoord2 0 (0::GLfloat))
                               vertex (Vertex2 x y)
                               texCoord (TexCoord2 1 (0::GLfloat))
                               vertex (Vertex2 (x+w) y)
                               texCoord (TexCoord2 1 (1::GLfloat))
                               vertex (Vertex2 (x+w) (y+h))
                               texCoord (TexCoord2 0 (1::GLfloat))
                               vertex (Vertex2 x (y+h))
