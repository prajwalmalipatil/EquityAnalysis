from enum import Enum
from dataclasses import dataclass
from typing import List

class TokenType(Enum):
    KEYWORD = "KEYWORD"           # e.g., "repo", "liquidity"
    FIELD_NAME = "FIELD_NAME"     # e.g., "confidence", "category"
    OPERATOR = "OPERATOR"         # e.g., ":", ">", "<", ">=", "<="
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    STRING = "STRING"             # e.g., "Liquidity Management" (in quotes)
    NUMBER = "NUMBER"             # e.g., 90
    EOF = "EOF"

@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    position: int

class LexerError(Exception):
    pass

class Lexer:
    """A lexical analyzer that converts a search query string into a stream of tokens."""
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None
        
    def advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
        
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
            
    def number(self) -> Token:
        result = ''
        start_pos = self.pos
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        return Token(TokenType.NUMBER, result, start_pos)
        
    def string(self) -> Token:
        start_pos = self.pos
        self.advance() # skip quote
        result = ''
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()
            
        if self.current_char == '"':
            self.advance() # skip closing quote
            
        return Token(TokenType.STRING, result, start_pos)
        
    def identifier(self) -> Token:
        result = ''
        start_pos = self.pos
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char in '_-'):
            result += self.current_char
            self.advance()
            
        value_upper = result.upper()
        if value_upper == "AND":
            return Token(TokenType.AND, value_upper, start_pos)
        elif value_upper == "OR":
            return Token(TokenType.OR, value_upper, start_pos)
        elif value_upper == "NOT":
            return Token(TokenType.NOT, value_upper, start_pos)
            
        # Is it a field name? (Followed by operator like : > < =)
        peek_pos = self.pos
        while peek_pos < len(self.text) and self.text[peek_pos].isspace():
            peek_pos += 1
            
        if peek_pos < len(self.text) and self.text[peek_pos] in (':', '>', '<', '='):
            return Token(TokenType.FIELD_NAME, result, start_pos)
            
        return Token(TokenType.KEYWORD, result, start_pos)
        
    def tokenize(self) -> List[Token]:
        tokens = []
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
                
            if self.current_char.isdigit():
                tokens.append(self.number())
                continue
                
            if self.current_char == '"':
                tokens.append(self.string())
                continue
                
            if self.current_char.isalpha() or self.current_char == '_':
                tokens.append(self.identifier())
                continue
                
            if self.current_char == ':':
                tokens.append(Token(TokenType.OPERATOR, ":", self.pos))
                self.advance()
                continue
                
            if self.current_char in ('>', '<', '='):
                start_pos = self.pos
                op = self.current_char
                self.advance()
                if self.current_char == '=':
                    op += '='
                    self.advance()
                tokens.append(Token(TokenType.OPERATOR, op, start_pos))
                continue
                
            if self.current_char == '(':
                tokens.append(Token(TokenType.LPAREN, '(', self.pos))
                self.advance()
                continue
                
            if self.current_char == ')':
                tokens.append(Token(TokenType.RPAREN, ')', self.pos))
                self.advance()
                continue
                
            raise LexerError(f"Invalid character at position {self.pos}: '{self.current_char}'")
            
        tokens.append(Token(TokenType.EOF, "", self.pos))
        return tokens
