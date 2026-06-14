from typing import List, Optional
from src.services.macro_intelligence.query.lexer import Token, TokenType
from src.services.macro_intelligence.query.ast import ASTNode, KeywordNode, FieldOpNode, NotNode, BinaryOpNode

class ParserError(Exception):
    pass

class Parser:
    """Parses a stream of tokens into an AST."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else None
        
    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
            
    def match(self, token_type: TokenType):
        if self.current_token and self.current_token.type == token_type:
            self.advance()
        else:
            actual = self.current_token.type if self.current_token else "None"
            raise ParserError(f"Expected {token_type}, got {actual}")
            
    def parse(self) -> Optional[ASTNode]:
        if not self.tokens or self.current_token.type == TokenType.EOF:
            return None
            
        node = self.expr()
        if self.current_token and self.current_token.type != TokenType.EOF:
            raise ParserError(f"Unexpected token at end: {self.current_token}")
        return node
        
    def expr(self) -> ASTNode:
        """expr: term (OR term)*"""
        node = self.term()
        
        while self.current_token and self.current_token.type == TokenType.OR:
            self.match(TokenType.OR)
            right = self.term()
            node = BinaryOpNode(operator="OR", left=node, right=right)
            
        return node
        
    def term(self) -> ASTNode:
        """term: factor ((AND)? factor)*"""
        node = self.factor()
        
        while self.current_token and self.current_token.type not in (TokenType.EOF, TokenType.OR, TokenType.RPAREN):
            # If explicit AND
            if self.current_token.type == TokenType.AND:
                self.match(TokenType.AND)
                right = self.factor()
                node = BinaryOpNode(operator="AND", left=node, right=right)
            else:
                # Implicit AND (e.g. "repo liquidity" -> repo AND liquidity)
                right = self.factor()
                node = BinaryOpNode(operator="AND", left=node, right=right)
                
        return node
        
    def factor(self) -> ASTNode:
        """factor: NOT factor | ( expr ) | FIELD_NAME OPERATOR VALUE | KEYWORD | STRING"""
        tok = self.current_token
        
        if tok.type == TokenType.NOT:
            self.match(TokenType.NOT)
            node = self.factor()
            return NotNode(operand=node)
            
        elif tok.type == TokenType.LPAREN:
            self.match(TokenType.LPAREN)
            node = self.expr()
            self.match(TokenType.RPAREN)
            return node
            
        elif tok.type == TokenType.FIELD_NAME:
            field_name = tok.value
            self.match(TokenType.FIELD_NAME)
            
            op_tok = self.current_token
            if op_tok.type != TokenType.OPERATOR:
                raise ParserError("Expected operator after field name")
            op = op_tok.value
            self.match(TokenType.OPERATOR)
            
            val_tok = self.current_token
            if val_tok.type in (TokenType.KEYWORD, TokenType.STRING, TokenType.NUMBER):
                val = val_tok.value
                # If number, convert it? We'll leave it as str for the AST and let Semantic Analyzer handle it.
                self.advance()
                return FieldOpNode(field=field_name, operator=op, value=val)
            else:
                raise ParserError("Expected value after field operator")
                
        elif tok.type == TokenType.KEYWORD or tok.type == TokenType.STRING or tok.type == TokenType.NUMBER:
            val = tok.value
            self.advance()
            return KeywordNode(value=val)
            
        raise ParserError(f"Unexpected token in factor: {tok}")
