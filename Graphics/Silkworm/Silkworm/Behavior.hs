module Silkworm.Behavior where
  
  -- Various behaviors of the GameObjects
  data Behavior = BeStationary
                | BeSimple
                | BeCrazy

  instance Show Behavior where
    show BeStationary = "Stationary Behavior"
    show BeSimple     = "Simple Behavior"
    show BeCrazy      = "Crazy Behavior"
  
  