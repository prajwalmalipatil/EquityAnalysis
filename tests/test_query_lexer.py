import pytest
from src.services.macro_intelligence.query.lexer import Lexer, TokenType, LexerError

def test_lexer_basic_keywords():
    lexer = Lexer("repo liquidity")
    tokens = lexer.tokenize()
    assert len(tokens) == 3 # repo, liquidity, EOF
    assert tokens[0].type == TokenType.KEYWORD
    assert tokens[0].value == "repo"
    assert tokens[1].type == TokenType.KEYWORD
    assert tokens[1].value == "liquidity"
    assert tokens[2].type == TokenType.EOF

def test_lexer_field_operators():
    lexer = Lexer("confidence>90 category:RBI")
    tokens = lexer.tokenize()
    
    # confidence, >, 90, category, :, RBI, EOF
    assert len(tokens) == 7
    
    assert tokens[0].type == TokenType.FIELD_NAME
    assert tokens[0].value == "confidence"
    
    assert tokens[1].type == TokenType.OPERATOR
    assert tokens[1].value == ">"
    
    assert tokens[2].type == TokenType.NUMBER
    assert tokens[2].value == "90"
    
    assert tokens[3].type == TokenType.FIELD_NAME
    assert tokens[3].value == "category"
    
    assert tokens[4].type == TokenType.OPERATOR
    assert tokens[4].value == ":"
    
    assert tokens[5].type == TokenType.KEYWORD
    assert tokens[5].value == "RBI"

def test_lexer_boolean_logic():
    lexer = Lexer("repo AND (policy OR liquidity) NOT forex")
    tokens = lexer.tokenize()
    
    # repo, AND, (, policy, OR, liquidity, ), NOT, forex, EOF
    assert len(tokens) == 10
    
    assert tokens[0].type == TokenType.KEYWORD
    assert tokens[1].type == TokenType.AND
    assert tokens[2].type == TokenType.LPAREN
    assert tokens[3].type == TokenType.KEYWORD
    assert tokens[4].type == TokenType.OR
    assert tokens[5].type == TokenType.KEYWORD
    assert tokens[6].type == TokenType.RPAREN
    assert tokens[7].type == TokenType.NOT
    assert tokens[8].type == TokenType.KEYWORD

def test_lexer_strings():
    lexer = Lexer('theme:"Liquidity Management"')
    tokens = lexer.tokenize()
    
    assert len(tokens) == 4
    assert tokens[0].type == TokenType.FIELD_NAME
    assert tokens[1].type == TokenType.OPERATOR
    assert tokens[2].type == TokenType.STRING
    assert tokens[2].value == "Liquidity Management"

def test_lexer_invalid():
    lexer = Lexer("repo & liquidity")
    with pytest.raises(LexerError):
        lexer.tokenize()

def test_lexer_unclosed_string():
    lexer = Lexer('theme:"Liquidity Management')
    with pytest.raises(SyntaxError):
        lexer.tokenize()

