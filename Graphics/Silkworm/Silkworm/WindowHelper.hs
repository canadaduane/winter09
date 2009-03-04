module Silkworm.WindowHelper (
    initWindow,
    KeySet,
    getPressedKeys,
    keyIsPressed
  ) where
  
  import Control.Monad (when)
  import qualified Data.Set as Set
  import Graphics.UI.GLFW (
    Key(..), KeyButtonState(..), getKey,
    WindowMode(..), initialize, openWindow,
    windowTitle, windowCloseCallback)
  import Graphics.Rendering.OpenGL (Size(..), ($=))
  import System.Exit (ExitCode(..), exitWith)
  import Silkworm.Constants (windowDimensions)
  
  -- Enhance the Key data type so it can be a member of a Set
  instance Ord Key where
    (<=) a b = (fromEnum a) <= (fromEnum b)
  
  -- | Open a window using GLFW to the dimensions set in Silkworm.Constants
  initWindow :: IO ()
  initWindow = do
    let size = (uncurry Size) windowDimensions
    
    -- Create the UI Window
    assertTrue initialize "Failed to initialize GLFW"
    assertTrue (openWindow size [] Window) "Failed to open a window"
    
    -- Set Window Parameters
    windowTitle $= "Silkworm"
    
    -- Close window nicely
    windowCloseCallback   $= exitWith ExitSuccess
    
    return ()
  
  -- | Asserts that an @IO@ action returns @True@, otherwise
  --   fails with the given message.
  assertTrue :: IO Bool -> String -> IO ()
  assertTrue act msg = do {b <- act; when (not b) (fail msg)}
  
  -------------------------------------------------------
  -- Key Helper Functions
  -------------------------------------------------------
  
  -- Create a Set of keys for the key helper functions
  type KeySet = Set.Set Key

  -- | Given a list of Keys, return a KeySet containing only those keys
  -- | that are currently pressed.
  getPressedKeys :: [Key] -> IO KeySet
  getPressedKeys keys = do
    keyStates <- mapM getKey keys
    return (Set.fromList $ keyList (zip keys keyStates))
    where
      isPressed x = (snd x) == Press
      keyList pairs = map fst (filter isPressed pairs)

  -- | Returns true if the Key is a member of KeySet
  keyIsPressed :: Key -> KeySet -> Bool
  keyIsPressed k keys = Set.member k keys
