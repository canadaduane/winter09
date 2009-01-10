import System.Random (newStdGen)
import Data.Array.Parallel.PArray (PArray, randomRs)
 
import DotP (dotp_wrapper)  -- import vectorised code
 
main :: IO ()
main
  = do 
      gen1 <- newStdGen
      gen2 <- newStdGen
      let v = randomRs n range gen1
          w = randomRs n range gen2
      print $ dotp_wrapper v w   -- invoke vectorised code and print the result
  where
    n     = 10000        -- vector length
    range = (-100, 100)  -- range of vector elements 

