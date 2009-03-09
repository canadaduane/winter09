module Silkworm.VRML where
  -- http://www.web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/part1/grammar.html
  -- http://legacy.cs.uu.nl/daan/download/parsec/parsec.html
  -- https://svn-agbkb.informatik.uni-bremen.de/Hets/trunk/Search/SPASS/DFGParser.hs
  import Text.ParserCombinators.Parsec
  import Text.ParserCombinators.Parsec.Prim
  import qualified Text.ParserCombinators.Parsec.Token as PT
  
  vrmlDef :: PT.LanguageDef st
  vrmlDef = PT.emptyDef
      { PT.commentLine     = "#"
      , PT.nestedComments  = False
      , PT.identStart      = letter
      , PT.identLetter     = alphaNum <|> oneOf "_"
      , PT.reservedNames   = ["DEF", "EXTERNPROTO", "IS",
                           "NULL", "PROTO", "ROUTE", "TO",
                           "TRUE", "FALSE", "USE",
                           "eventIn", "eventOut",
                           "exposedField", "field"]
      , PT.caseSensitive   = True
      }
  
  lexer :: PT.TokenParser st
  lexer = PT.makeTokenParser vrmlDef
  
  comma      = PT.comma      lexer
  brackets   = PT.brackets   lexer
  natural    = PT.natural    lexer
  whiteSpace = PT.whiteSpace lexer
