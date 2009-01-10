{-# LANGUAGE PArr, ParallelListComp #-}
{-# OPTIONS -fvectorise -fdph-seq #-}
 
module DotP (dotp_double,dotp_wrapper)
where
 
import qualified Prelude
import Data.Array.Parallel.Prelude
import Data.Array.Parallel.Prelude.Double
 
dotp_double :: [:Double:] -> [:Double:] -> Double
dotp_double xs ys = sumP [:x * y | x <- xs | y <- ys:]
 
dotp_wrapper :: PArray Double -> PArray Double -> Double
{-# NOINLINE dotp_wrapper #-}
dotp_wrapper v w = dotp_double (fromPArrayP v) (fromPArrayP w)

