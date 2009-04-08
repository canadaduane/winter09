module Silkworm.Constants where
  
  import GHC.Int (Int32)
  import Physics.Hipmunk
  
  windowDimensions :: (Int32, Int32)
  windowDimensions = (800, 600)
  
  elasticity :: Float
  elasticity = 5.0
  
  -- | Strength of the gravitational force
  gravity :: Vector
  gravity = Vector 0 (-5)
  
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
  frameDelta :: Time
  frameDelta = 3.33e-3
  
  -- | How much slower should the slow mode be.
  slowdown :: Double
  slowdown = 10
  