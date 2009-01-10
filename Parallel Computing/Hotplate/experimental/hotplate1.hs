module Main where
import Array

newtype Hotplate = MakeHotplate (Array (Int, Int) Float)

instance Show Hotplate where
  show hp =
    rollDown hp 1
    where
      rollRight :: Hotplate -> Int -> Int -> String
      rollRight hp@(MakeHotplate a) x y
        | inRange (bounds a) (x, y) =
            show (a!(x, y)) ++ " " ++
            (rollRight hp (x + 1) y)
        | otherwise =
            "\n"
      
      rollDown :: Hotplate -> Int -> String
      rollDown hp@(MakeHotplate a) y
        | inRange (bounds a) (1, y) =
          (rollRight hp 1 y) ++
          (rollDown hp (y + 1))
        | otherwise =
          ""
      
makeHotplate :: Int -> Int -> Hotplate
makeHotplate w h = MakeHotplate
  (array ((1, 1), (w, h))
    [ ((x, y), 0) | x <- range(1,w),
                    y <- range(1,h) ])

-- hpSet :: Hotplate -> Int -> Int -> Float -> IO Hotplate
-- hpSet hp x y val =
--   do {
--     
--   }
  

hotplate = makeHotplate 3 3

main :: IO ()
main = do {
  putStr ("Hotplate: " ++ show hotplate);
  return ()
}

-- module Main where
-- import Control.Monad
-- import Control.Concurrent
-- import Control.Concurrent.STM
--  
-- main = do shared <- atomically $ newTVar 0
--           before <- atomRead shared
--           putStrLn $ "Before: " ++ show before
--           forkIO $ 25 `timesDo` (dispVar shared >> milliSleep 20)
--           forkIO $ 10 `timesDo` (appV ((+) 2) shared >> milliSleep 50)
--           forkIO $ 20 `timesDo` (appV pred shared >> milliSleep 25)
--           milliSleep 800
--           after <- atomRead shared
--           putStrLn $ "After: " ++ show after
--  where timesDo = replicateM_
--        milliSleep = threadDelay . (*) 1000
--  
-- atomRead = atomically . readTVar
-- dispVar x = atomRead x >>= print
-- appV fn x = atomically $ readTVar x >>= writeTVar x . fn