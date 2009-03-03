module Silkworm.Init (State, initialState) where
  
  import Data.Map
  import Physics.Hipmunk
  import Silkworm.Constants
  -- import Const
  
  data State = State {
    stSpace  :: Space,
    stShapes :: Map Shape (
      IO () {- Drawing -},
      IO () {- Removal -}
    )}
  
  initialState :: IO State
  initialState = do
    space  <- newSpace
    setGravity space gravity
    
    generateRandomGround space
    
    return (State space shapes)
    
    where
      shapes = fromList []
  
  generateRandomGround :: Space -> IO ()
  generateRandomGround space =
    return ()
    