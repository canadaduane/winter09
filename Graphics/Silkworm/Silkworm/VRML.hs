module VRML where
  import Prelude hiding (lex)
  import Text.ParserCombinators.Parsec
  import Text.ParserCombinators.Parsec.Token hiding (
    whiteSpace, lexeme, symbol, natural, braces, semi,
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
                 "eventIn", "eventOut", "exposedField", "field"],
              caseSensitive   = True
            }
  
  whiteSpace = T.whiteSpace lexer
  lexeme     = T.lexeme     lexer
  symbol     = T.symbol     lexer
  natural    = T.natural    lexer
  braces     = T.braces     lexer
  semi       = T.semi       lexer
  identifier = T.identifier lexer
  reserved   = T.reserved   lexer
  reservedOp = T.reservedOp lexer
  
  -- http://www.web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/part1/grammar.html
  
  vrmlScene  = do es <- many1 statements
                  eof
                  return es
  
  statements :: Parser String
  statements     = nodeStatement <|> protoStatement <|> routeStatement
  
  nodeStatement :: Parser String
  nodeStatement  = do reserved "DEF"
                      nodeNameId
                      node
               <|> do reserved "USE"
                      nodeNameId
               <|> node
  
  nodeNameId :: Parser String
  nodeNameId     = do identifier
  
  nodeTypeId :: Parser String
  nodeTypeId     = do identifier

  node :: Parser String
  node           = do reserved "Script"
                      braces scriptBody
               <|> do nodeTypeId
                      braces nodeBody
  
  nodeBody :: Parser String
  nodeBody = string "nodeBody"
  
  scriptBody :: Parser String
  scriptBody = string "scriptBody"
  
  protoStatement :: Parser String
  protoStatement = string "protoStatement"
  
  routeStatement :: Parser String
  routeStatement = string "routeStatement"
  
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
  