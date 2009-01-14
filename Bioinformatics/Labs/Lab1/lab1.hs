module Main where

sim a b
  | a == b     = 1
  | otherwise  = -1

wunsch :: String -> String -> Int
wunsch a b =

main = do
  putStrLn ("Similarity of A, A : " ++ (show (sim "A" "A")))