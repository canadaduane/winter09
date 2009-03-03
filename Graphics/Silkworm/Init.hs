module Silkworm.Init where
  
  import Physics.Hipmunk as H
  
  -- | Strength of the gravitational force
  gravity :: H.Vector
  gravity = H.Vector 0 (-800)
  
  -- | Desired (and maximum) frames per second.
  desiredFPS :: Double
  desiredFPS = 60.0

  -- | How much seconds a frame lasts.
  framePeriod :: Double
  framePeriod = 1.0 / desiredFPS

  -- | How many steps should be done per frame.
  frameSteps :: Double
  frameSteps = 6

  -- | Maximum number of steps per frame (e.g. if lots of frames get
  --   dropped because the window was minimized)
  maxSteps :: Int
  maxSteps = 20

  -- | How much time should pass in each step.
  frameDelta :: H.Time
  frameDelta = 3.33e-3
  