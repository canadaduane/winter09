module Silkworm.WaveFront (readWaveFront, Object3D(..)) where
  
  import Data.List (groupBy)
  import Silkworm.Object3D
  
  type IndexTriple = (Int, Int, Int)
  
  data ObjData = MaterialFile String
               | ObjName String
               | GrpName String
               | Vertex VectorTriple
               | Normal VectorTriple
               | FaceIndex [IndexTriple]
               | NoData
    deriving Show
  
  fromVertex :: ObjData -> VectorTriple
  fromVertex (Vertex d) = d

  fromNormal :: ObjData -> VectorTriple
  fromNormal (Normal d) = d

  fromFaceIndex :: ObjData -> [IndexTriple]
  fromFaceIndex (FaceIndex d) = d
  
  -- Exported function, reads a WaveFront "obj" formatted file
  readWaveFront :: String -> Object3D
  readWaveFront = reconstitute . parse
  
  reconstitute :: [ObjData] -> Object3D
  reconstitute ds = Object3D "obj" faces
    where vertices         = map fromVertex    $ filter isVertex ds
          normals          = map fromNormal    $ filter isNormal ds
          faceIdxs         = map fromFaceIndex $ filter isFace   ds
          faces            = map (map vnPair) faceIdxs
          vnPair (v, t, n) = (vertices !! (v - 1), normals  !! (n - 1))
          -- Filters for ObjData types
          isVertex (Vertex d)    = True
          isVertex _             = False
          isNormal (Normal d)    = True
          isNormal _             = False
          isFace   (FaceIndex d) = True
          isFace   _             = False
  
  parse :: String -> [ObjData]
  parse s = map readLine ls
    where
      ls = lines $ stripComments s
      readLine line =
        let ws = words $ dropWhile (== ' ') line
        in toData ws
      toData []     = NoData
      toData (w:ws) =
        let name  = (foldr (++) [] ws)
            nums  = map read ws
            faces = map indexUnslash ws
        in case w of
             "mtllib" -> MaterialFile name
             "o"      -> ObjName name
             "g"      -> GrpName name
             "v"      -> Vertex (triple nums)
             "vn"     -> Normal (triple nums)
             "f"      -> FaceIndex faces
             _        -> NoData
        
  
  -- Remove inline '#' comments
  stripComments :: String -> String
  stripComments []     = []
  stripComments (s:ss) =
    if s == comment
      then stripComments $ erase ss
      else s : stripComments ss
    where
      eol = '\n'
      comment = '#'
      erase [] = []
      erase (s:ss) =
        if s == eol
          then (s:ss)
          else erase ss
  
  -- Split a string on a particular character, being careful to return the
  -- expected number of parts 
  splitChar :: Char -> String -> [String]
  splitChar char ss = map removeChar $ groupBySlash ('/':ss)
    where
      removeChar   = dropWhile (== char)
      groupBySlash = groupBy (\a b -> b /= char)
  
  triple :: [a] -> (a, a, a)
  triple z = (z !! 0, z !! 1, z !! 2)
  
  -- Take a slash-separated string of 3 integers and return them in a triple
  -- e.g. "1/2/3" => (1, 2, 3)
  indexUnslash :: String -> IndexTriple
  indexUnslash = triple . map safeRead . splitChar '/'
    where safeRead n = if n == "" then 0 else read n

  