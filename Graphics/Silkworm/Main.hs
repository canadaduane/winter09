module Main (main) where

  import Physics.Hipmunk (initChipmunk)
  import Silkworm.WindowHelper (initWindow)
  import Silkworm.OpenGLHelper (initOpenGL)
  import Silkworm.Title (showTitleScreen)
  import Silkworm.Game (startGame)
  
  main :: IO ()
  main = do
    initializeSilkworm
    screenLoop
  
  screenLoop :: IO ()
  screenLoop = do
    showTitleScreen
    startGame
    screenLoop
  
  initializeSilkworm :: IO ()
  initializeSilkworm = do
    -- Open the game window
    initWindow

    -- Prepare OpenGL
    initOpenGL

    -- Start physics engine
    initChipmunk