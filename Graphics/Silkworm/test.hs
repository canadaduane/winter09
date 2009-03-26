module Test where
  
  import Data.List (groupBy)
  
  obj_file = "silkworm.obj"
  
  type Vector = (Double, Double, Double)
  type IndexTriple = (Int, Int, Int)
  data Face = Face [Vector] [Vector]
  data Object = Object String [Face]
  
  data ObjData = MaterialFile String
               | ObjName String
               | GrpName String
               | Vertex Vector
               | Normal Vector
               | FaceIndex [IndexTriple]
               | NoData
    deriving Show
  
  parse :: String -> [ObjData]
  parse s = map readLine ls
    where
      ls = lines $ stripComments s
      readLine line =
        let ws = words $ dropWhile (== ' ') line
        in toData ws
      toData []     = NoData
      toData (w:ws) =
        let name = (foldr (++) [] ws)
            nums = map read ws
        in case w of
             "mtllib" -> MaterialFile name
             "o"      -> ObjName name
             "v"      -> Vertex (triple nums)
             "n"      -> Normal (triple nums)
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
  splitChar char ss = 
    case ss of
      ns | ns == allChar -> replicate (1 + length ss) ""
      otherwise          -> map removeChar $ groupBySlash ss
    where
      removeChar        = dropWhile (== char)
      groupBySlash xs   = groupBy (\a b -> b /= char) xs
      allChar           = replicate (length ss) char
  
  triple :: [a] -> (a, a, a)
  triple z = (z !! 0, z !! 1, z !! 2)
  
  -- Take a slash-separated string of 3 integers and return them in a triple
  -- e.g. "1/2/3" => (1, 2, 3)
  indexUnslash :: String -> IndexTriple
  indexUnslash = triple . map safeRead . splitChar '/'
    where safeRead n = if n == "" then 0 else read n