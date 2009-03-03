module Main (main) where

  import Data.IORef
  import Silkworm.Init
  import Silkworm.Game
  
  main :: IO ()
  main = do
    stateRef <- initialize
    startGame stateRef