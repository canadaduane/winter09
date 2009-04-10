module Silkworm.Substance where
  
  import Physics.Hipmunk

  data Substance = Substance
                   Elasticity
                   Friction  
                   Float       -- Density
    deriving Show
  
  genericSubstance = Substance 0.1 0.1 0.5
  rubber           = Substance 0.8 1.0 0.5
  wood             = Substance 0.3 0.8 0.5
  concrete         = Substance 0.1 0.8 0.8
  