{-# OPTIONS_GHC -XBangPatterns #-}

module Silkworm.Math where
  
  import Data.List (transpose, foldl')
  
  -- | Cross product of two 3-dimensional tuples
  cross :: (Floating a) => (a, a, a) -> (a, a, a) -> (a, a, a)
  cross (a1, a2, a3) (b1, b2, b3) = ((a2*b3 - a3*b2), (a3*b1 - a1*b3), (a1*b2 - a2*b1))
  
  dot :: (Floating a) => (a, a, a) -> (a, a, a) -> a
  dot (a1, a2, a3) (b1, b2, b3) = a1*b1 + a2*b2 + a3*b3
  
  -- | Numerically stable mean from Math.Statistics
  average :: Floating a => [a] -> a
  average x = fst $ foldl' (\(!m, !n) x -> (m+(x-m)/(n+1),n+1)) (0,0) x
  
  distance3d :: (Floating a) => (a, a, a) -> (a, a, a) -> a
  distance3d (ax, ay, az) (bx, by, bz) = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    where dx = bx - ax
          dy = by - ay
          dz = bz - az
  
  times n (x, y, z) = (n*x, n*y, n*z)
  divideby vt n = times (1/n) vt
  minus (x1, y1, z1) (x2, y2, z2) = (x1-x2, y1-y2, z1-z2)
  plus  (x1, y1, z1) (x2, y2, z2) = (x1+x2, y1+y2, z1+z2)

  -- newtype Matrix a = Matrix [[a]] deriving (Eq, Show)
  -- 
  -- instance Num a => Num (Matrix a) where
  --    Matrix as + Matrix bs = Matrix (zipWith (zipWith (+)) as bs)
  --    Matrix as - Matrix bs = Matrix (zipWith (zipWith (-)) as bs)
  --    Matrix as * Matrix bs =
  --       Matrix [[sum $ zipWith (*) a b | b <- transpose bs] | a <- as]
  --    negate (Matrix as) = Matrix (map (map negate) as)
  --    fromInteger x = Matrix (iterate (0:) (fromInteger x : repeat 0))
  --    abs m = m
  --    signum _ = 1
  -- 
  -- apply :: Num a => Matrix a -> [a] -> [a]
  -- apply (Matrix as) b = [sum (zipWith (*) a b) | a <- as]
  
  