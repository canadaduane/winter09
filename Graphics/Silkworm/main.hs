module Main (main) where

  import Physics.Hipmunk (initChipmunk)
  import Silkworm.WindowHelper (initWindow)
  import Silkworm.OpenGLHelper (initOpenGL)
  import Silkworm.Title (showTitleScreen)
  import Graphics.UI.GLFW
  import Graphics.Rendering.OpenGL
  
  main :: IO ()
  main = do
    initializeSilkworm
    v <- get (windowParam DepthBits)
    putStrLn $ show v
    showTitleScreen
  
  initializeSilkworm :: IO ()
  initializeSilkworm = do
    -- Open the game window
    initWindow

    -- Prepare OpenGL
    initOpenGL

    -- Start physics engine
    initChipmunk