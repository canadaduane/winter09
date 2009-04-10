module Silkworm.Behavior where
  
  -- Various behaviors of the GameObjects
  data Behavior = BeStationary
                | BeSimple
                | BeCrazy
                | BeControllable
    deriving Eq

  instance Show Behavior where
    show BeStationary   = "Stationary Behavior"
    show BeSimple       = "Simple Behavior"
    show BeCrazy        = "Crazy Behavior"
    show BeControllable = "Controllable Behavior"
  
  