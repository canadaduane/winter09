module VRML where
  import Prelude hiding (lex)
  import Text.ParserCombinators.Parsec
  import Text.ParserCombinators.Parsec.Token hiding (
    whiteSpace, lexeme, symbol, natural, float, braces, semi,
    identifier, reserved, reservedOp)
  import qualified Text.ParserCombinators.Parsec.Token as T
  import Text.ParserCombinators.Parsec.Language (emptyDef)
  
  lexer :: TokenParser ()
  lexer = makeTokenParser
            emptyDef {
              commentLine     = "#",
              identStart      = letter,
              identLetter     = alphaNum <|> oneOf "_",
              reservedOpNames =
                ["DEF", "EXTERNPROTO", "FALSE", "IS", "NULL",
                 "PROTO", "ROUTE", "TO", "TRUE", "USE", "Script",
                 "eventIn", "eventOut", "exposedField", "field",
                 "MFColor", "MFFloat", "MFInt32",
                 "MFNode", "MFRotation", "MFString",
                 "MFTime", "MFVec2f", "MFVec3f",
                 "SFBool", "SFColor", "SFFloat",
                 "SFImage", "SFInt32", "SFNode",
                 "SFRotation", "SFString", "SFTime",
                 "SFVec2f", "SFVec3f"],
              caseSensitive   = True
            }
  
  whiteSpace = T.whiteSpace lexer
  lexeme     = T.lexeme     lexer
  symbol     = T.symbol     lexer
  natural    = T.natural    lexer
  float      = T.float      lexer
  braces     = T.braces     lexer
  semi       = T.semi       lexer
  identifier = T.identifier lexer
  reserved   = T.reserved   lexer
  reservedOp = T.reservedOp lexer
  
  -- http://www.web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/part1/grammar.html
  
  vrmlScene :: Parser String
  vrmlScene            = do es <- many1 statement
                            eof
                            return es
  
  statement :: Parser String
  statement            = nodeStatement <|> protoStatement <|> routeStatement
  
  nodeStatement :: Parser String
  nodeStatement        = do reserved "DEF"
                            nodeNameId
                            node
                     <|> do reserved "USE"
                            nodeNameId
                     <|> node
  
  protoStatement :: Parser String
  protoStatement       = proto <|> externProto
  
  proto :: Parser String
  proto                = do reserved "PROTO"
                            nodeTypeId
                            brackets (many1 interfaceDeclaration)
                            braces protoBody
  
  protoBody :: Parser String
  protoBody            = do many1 protoStatement
                            many statement
  
  interfaceDeclaration :: Parser String
  interfaceDeclaration = do reserved "exposedField"
                            fieldType
                            fieldId
                            fieldValue
                     <|> do restrictDeclaration
  
  restrictDeclaration :: Parser String
  restrictDeclaration  = do reserved "eventIn"
                            fieldType
                            eventInId
                     <|> do reserved "eventOut"
                            fieldType
                            eventOutId
                     <|> do reserved "field"
                            fieldType
                            fieldId
                            fieldValue
  
  externProto :: Parser String
  externProto          = do reserved "EXTERNPROTO"
                            nodeTypeId
                            brackets (many1 externDeclaration)
                            urlList
  
  externDeclaration    = do reserved "eventIn"
                            fieldType
                            eventInId
                     <|> do reserved "eventOut"
                            fieldType
                            eventOutId
                     <|> do reserved "field"
                            fieldType
                            fieldId
                     <|> do reserved "exposedField"
                            fieldType
                            fieldId
  
  urlList :: Parser String
  urlList              = stringLiteral
  
  nodeNameId :: Parser String
  nodeNameId           = identifier
  
  nodeTypeId :: Parser String
  nodeTypeId           = identifier
  
  node :: Parser String
  node                 = do reserved "Script"
                            braces (many scriptBodyElement)
                     <|> do identifier
                            braces (many nodeBodyElement)
  
  scriptBodyElement :: Parser String
  scriptBodyElement    = try do reserved "eventIn" <|> reserved "eventOut" <|> reserved "field"
                                fieldType
                                identifier
                                reserved "IS"
                                identifier
                     <|> restrictDeclaration
                     <|> nodeBodyElement
  
  nodeBodyElement :: Parser String
  nodeBodyElement      = try do identifier
                                reserved "IS"
                                identifier
                     <|> do fieldId
                            fieldValue
                     <|> routeStatement
                     <|> protoStatement
  
  routeStatement :: Parser String
  routeStatement = string "routeStatement"
  
  fieldType :: Parser String
  fieldType            = foldr1 (<|>) (map reserved types)
    where types = ["MFColor", "MFFloat", "MFInt32",
                   "MFNode", "MFRotation", "MFString",
                   "MFTime", "MFVec2f", "MFVec3f",
                   "SFBool", "SFColor", "SFFloat",
                   "SFImage", "SFInt32", "SFNode",
                   "SFRotation", "SFString", "SFTime",
                   "SFVec2f", "SFVec3f"],
  
  fieldValue :: Parser String
  fieldValue           = sfboolValue
                     <|> sfcolorValue
                     <|> sffloatValue
                     <|> sfimageValue
                     <|> sfint32Value
                     <|> sfnodeValue
                     <|> sfrotationValue
                     <|> sfstringValue
                     <|> sfvec2fValue
                     <|> sfvec3fValue
                     <|> mfcolorValue
                     <|> mffloatValue
                     <|> mfint32Value
                     <|> mfnodeValue
                     <|> mfrotationValue
                     <|> mfstringValue
                     <|> mftimeValue
                     <|> mfvec2fValue
                     <|> mfvec3fValue
  
  mftimeValue     = brackets (many float)
  sfboolValue     = reserved "TRUE" <|> reserved "FALSE"
  sfnodeValue     = reserved "NULL" <|> nodeStatement
  sfstringValue   = stringLiteral
  sf4fValue       = float >> whiteSpace >> float >> whiteSpace >> float >> whiteSpace >> float
  sf3fValue       = float >> whiteSpace >> float >> whiteSpace >> float
  sf2fValue       = float >> whiteSpace >> float
  sf1fValue       = float
  sf3iValue       = natural >> whiteSpace >> natural >> whiteSpace >> natural
  sf1iValue       = natural

  lex :: String -> IO ()
  lex input = runLex vrmlScene input
  
  runLex :: Show a => Parser a -> String -> IO () 
  runLex p input = run top input
    where top = do whiteSpace 
                   x <- p 
                   eof
                   return x
  
  run :: Show a => Parser a -> String -> IO () 
  run p input = case (parse p "" input) of 
                  Left  err -> putStr "parse error at " >> print err
                  Right ret -> print ret
  