module Main (main) where

  import Physics.Hipmunk (initChipmunk)
  import Silkworm.WindowHelper (initWindow)
  import Silkworm.OpenGLHelper (initOpenGL)
  import Silkworm.Title (showTitleScreen)
  
  main :: IO ()
  main = do
    initializeSilkworm
    showTitleScreen
  
  initializeSilkworm :: IO ()
  initializeSilkworm = do
    -- Open the game window
    initWindow

    -- Prepare OpenGL
    initOpenGL

    -- Start physics engine
    initChipmunk
  