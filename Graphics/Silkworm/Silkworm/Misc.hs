module Silkworm.Misc (assertTrue) where
  
  import Control.Monad (when)
  
  -- | Asserts that an @IO@ action returns @True@, otherwise
  --   fails with the given message.
  assertTrue :: IO Bool -> String -> IO ()
  assertTrue act msg = do {b <- act; when (not b) (fail msg)}
