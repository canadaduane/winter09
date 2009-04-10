module Silkworm.Action where
  
  -- Named IO Actions (for debugging and display)
  data Action = Action String (IO ())
  instance Show Action where
    show (Action name _) = name
  
  act :: Action -> IO ()
  act (Action name fn) = fn
  
  -- Special action that wraps a function in an "Anonymous" Action
  action :: IO () -> Action
  action fn = Action "Anonymous" fn
  
  -- Special action that does nothing (used for default values and tests)
  inaction :: Action
  inaction = Action "Inaction" nothing
    where nothing = return ()
  
  