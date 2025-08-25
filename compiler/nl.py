#!/usr/bin/env python3
# newlang_compiler.py - Modular compiler with extensible architecture

import re
import sys
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any, Union, Dict, Type, Callable, Set
from abc import ABC, abstractmethod

# ==================== BASE COMPONENTS ====================

class CompilerComponent(ABC):
    """Base class for all compiler components."""
    
    def __init__(self, compiler: 'Compiler'):
        self.compiler = compiler

class ASTNode(ABC):
    """Base class for all AST nodes."""
    pass

# ==================== TOKENIZATION ====================

@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int

class TokenSpec:
    """Container for token specifications that can be extended."""
    
    def __init__(self):
        self.specs = [
            # Keywords
            ('KW_DECLARE',   r'\bdeclare\b'),
            ('KW_DEF',       r'\bdef\b'),
            ('KW_EXTEND',    r'\bextend\b'),
            ('KW_TYPE',      r'\btype\b'),
            ('KW_STRUCT',    r'\bstruct\b'),
            ('KW_AUTO',      r'\bauto\b'),
            ('KW_CONST',     r'\bconst\b'),
            ('KW_FINALIZE',  r'\bfinalize\b'),
            ('KW_PTR',       r'\bptr\b'),
            ('KW_PRIVATE',   r'\bprivate\b'),
            ('KW_MAIN',      r'\b__main__\b'),
            ('KW_NEW',       r'\bnew\b'),
            ('KW_IO',        r'\bio\b'),
            ('KW_IMPORT',    r'\bimport\b'),
            ('KW_AS',        r'\bas\b'),
            ('KW_FROM',      r'\bfrom\b'),
            ('KW_INTO',      r'\binto\b'),
            ('KW_FOR',       r'\bfor\b'),
            ('KW_WHILE',     r'\bwhile\b'),
            ('KW_OR',        r'\bor\b'),
            ('KW_AND',       r'\band\b'),
            ('KW_IF',        r'\bif\b'),
            ('KW_ELSE',      r'\belse\b'),
            ('KW_END',       r'\bend\b'),
            ('KW_CONT',      r'\bcont\b'),
            ('KW_LAMBDA',    r'\blambda\b'),
            ('KW_REASSIGN',  r'\breassign\b'),
            ('KW_NIL',       r'\bnil\b'),

            # Operators and symbols
            ('EQEQ',         r'=='),
            ('NEQ',          r'!='),
            ('LE',           r'<='),
            ('GE',           r'>='),
            ('ARROW',        r'->'),
            ('SCOPE',        r':'),
            
            # Literals
            ('NUMBER',       r'\d+(\.\d+)?'),
            ('STRING',       r'"[^"\\]*(?:\\.[^"\\]*)*"|“[^”\\]*(?:\\.[^”\\]*)*”'),
            
            # Comments
            ('LINE_COMMENT', r'//[^\n]*'),
            ('BLOCK_COMMENT',r'/\*(?s:.*?)\*/'),
            
            # Identifiers (must be after keywords)
            ('IDENT',        r'[A-Za-z_][A-Za-z0-9_]*'),
            
            # Other
            ('OP',           r'[+\-*/=]'),
            ('SYMBOL',       r'[{}();,.\[\]<>]'),
            ('WS',           r'[ \t]+'),
            ('NEWLINE',      r'\n'),
            ('AMP',          r'&'),
            ('MISMATCH',     r'.'),
        ]
    
    def add_token(self, name: str, pattern: str):
        """Add a new token type to the specification."""
        self.specs.append((name, pattern))
    
    def remove_token(self, name: str):
        """Remove a token type from the specification."""
        self.specs = [(n, p) for n, p in self.specs if n != name]
    
    def get_regex(self):
        """Compile the token specification into a regex pattern."""
        return re.compile('|'.join(f'(?P<{k}>{p})' for k, p in self.specs))

class Tokenizer(CompilerComponent):
    """Handles tokenization of source code."""
    
    def __init__(self, compiler: 'Compiler'):
        super().__init__(compiler)
        self.token_spec = TokenSpec()
    
    def tokenize(self, source: str) -> List[Token]:
        tokens: List[Token] = []
        line = 1
        col = 1
        regex = self.token_spec.get_regex()
        
        for m in regex.finditer(source):
            kind = m.lastgroup
            value = m.group()
            
            if kind == 'NEWLINE':
                line += 1
                col = 1
                continue
                
            if kind in ('WS', 'LINE_COMMENT', 'BLOCK_COMMENT'):
                col += len(value)
                continue
                
            if kind == 'MISMATCH':
                raise SyntaxError(f"Unexpected character {value!r} at line {line}, col {col}")
                
            tokens.append(Token(kind, value, line, col))
            col += len(value)
            
        return tokens

# ==================== ABSTRACT SYNTAX TREE ====================

@dataclass
class Program(ASTNode):
    items: List[ASTNode]

@dataclass
class FuncDecl(ASTNode):
    name: str
    params: List[str]
    ret: Optional[str]

@dataclass
class VarDecl(ASTNode):
    is_const: bool
    type_name: Optional[str]
    name: str
    init: ASTNode

@dataclass
class FuncDef(ASTNode):
    is_main: bool
    name: str
    params: List[str]
    ret: Optional[str]
    body: List[ASTNode]

@dataclass
class Call(ASTNode):
    name: str
    args: List[ASTNode]

@dataclass
class MemberCall(ASTNode):
    receiver: ASTNode
    member: str
    args: List[ASTNode]

@dataclass
class VarRef(ASTNode):
    name: str

@dataclass
class AddressOf(ASTNode):
    target: VarRef

@dataclass
class NumberLiteral(ASTNode):
    value: Union[int, float]

@dataclass
class StringLiteral(ASTNode):
    value: str

@dataclass
class BinaryExpr(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

@dataclass
class Finalize(ASTNode):
    expr: ASTNode

@dataclass
class MemberAssign(ASTNode):
    target_object: str
    field: str
    expr: ASTNode

@dataclass
class FieldDecl(ASTNode):
    is_private: bool
    type_name: str
    name: str
    default: Optional[ASTNode]

@dataclass
class TypeStruct(ASTNode):
    name: str
    fields: List[FieldDecl]
    ctor_params: List[str]
    base_name: Optional[str] = None

@dataclass
class ExtendFunc(ASTNode):
    name: str
    params: List[str]
    ret: Optional[str]
    body: List[ASTNode]

@dataclass
class ExtendCtor(ASTNode):
    type_name: str
    params: List[str]
    body: List[ASTNode]
    base_delegate: Optional[tuple] = None

@dataclass
class QualifiedRef(ASTNode):
    alias: str
    name: str

@dataclass
class CallExpr(ASTNode):
    callee: ASTNode
    args: List[ASTNode]

@dataclass
class NewLiteralToArray(ASTNode):
    text: str

@dataclass
class ForStmt(ASTNode):
    init: ASTNode
    cond: ASTNode
    step: ASTNode
    body: List[ASTNode]

@dataclass
class WhileStmt(ASTNode):
    cond: ASTNode
    body: List[ASTNode]

@dataclass
class Assign(ASTNode):
    target: ASTNode
    expr: ASTNode

@dataclass
class YieldStmt(ASTNode):
    expr: ASTNode

@dataclass
class IfStmt(ASTNode):
    cond: ASTNode
    then_body: List[ASTNode]
    else_body: Optional[List[ASTNode]] = None

@dataclass
class BreakStmt(ASTNode):
    condition: Optional[ASTNode] = None

@dataclass
class ArrayIndex(ASTNode):
    array: ASTNode
    index: ASTNode

@dataclass
class ArrayAssignment(ASTNode):
    array: ASTNode
    index: ASTNode
    value: ASTNode

@dataclass
class NewArray(ASTNode):
    size: ASTNode

@dataclass
class NewObject(ASTNode):
    type_name: str
    args: List[ASTNode]

@dataclass
class ContinueStmt(ASTNode):
    pass

@dataclass
class IteratorDef(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]

@dataclass
class ModuleSymbols:
    types: Set[str]
    funcs: Set[str]

@dataclass
class LambdaLiteral(ASTNode):
    params: List[str]           # kept for future; we run with implicit args
    ret: Optional[str]
    body: List[ASTNode]

@dataclass
class LambdaWrapper(ASTNode):
    captures: List[ASTNode]     # expressions -> values at add-time
    bound: bool                 # (&) present
    func: Union['LambdaLiteral', VarRef]  # def {...} or a named function

@dataclass
class NewDelegateCollection(ASTNode):
    bound: bool                 # from (&)
    default: Optional[LambdaWrapper]

@dataclass
class Reassign(ASTNode):
    target: Any
    value: Any

@dataclass
class NilLiteral(ASTNode):
    pass

@dataclass
class ExtendMethod(ASTNode):
    type_name: str
    method: str
    params: List[str]
    ret: Optional[str]
    body: List[ASTNode]

@dataclass
class FieldGet(ASTNode):
    receiver: ASTNode
    field: str

@dataclass
class ImportStmt(ASTNode):
    path: str
    alias: Optional[str] = None

# ==================== EXCEPTION CLASSES ====================

class _BreakSignal(Exception):
    pass

class _ContinueSignal(Exception):
    pass

class _ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

class SemanticError(Exception):
    pass

# ==================== PARSING ====================

class Parser(CompilerComponent):
    """Handles parsing of tokens into an AST."""
    
    def __init__(self, compiler: 'Compiler'):
        super().__init__(compiler)
        self.tokens: List[Token] = []
        self.i = 0
        
    def parse(self, tokens: List[Token]) -> Program:
        self.tokens = tokens
        self.i = 0
        return self.parse_program()
    
    def parse_program(self) -> Program:
        items = []
        while self.peek():
            # Try each parsing method in order
            for parser_method in self.get_program_parsers():
                result = parser_method()
                if result is not None:
                    items.append(result)
                    break
            else:
                # No parser matched, skip token
                self.i += 1
                
        return Program(items)
    
    def get_program_parsers(self) -> List[Callable]:
        """Return a list of parser methods for program-level constructs."""
        return [
            self.try_parse_import,
            self.try_parse_declare,
            self.try_parse_const,
            self.try_parse_main_def,
            self.try_parse_def,
            self.try_parse_extend,
        ]
    
    # --- Parser.try_parse_import ---
    def try_parse_import(self) -> Optional[ImportStmt]:
        if not self.match('KW_IMPORT'): return None
        self.expect('SYMBOL')  # '{'
        path = self.expect('STRING').value.strip('"').strip('“').strip('”')
        self.expect('SYMBOL')  # '}'
        alias = None
        if self.match('KW_AS'):
            alias = self.expect('IDENT').value
        if self.peek() and self.peek().value == ';':
            self.i += 1
        return ImportStmt(path=path, alias=alias)



    
    def try_parse_declare(self) -> Optional[ASTNode]:
        if not self.match('KW_DECLARE'):
            return None
            
        if self.match('KW_DEF'):
            return self.parse_func_decl()
        elif self.match('KW_TYPE'):
            return self.parse_type_decl()
        else:
            raise SyntaxError("Expected 'def' or 'type' after 'declare'")
    
    def try_parse_const(self) -> Optional[VarDecl]:
        if not self.match('KW_CONST'):
            return None
        # support: const lambda NAME [= expr]?;
        t0 = self.expect('IDENT')
        if t0.value == 'lambda':
            type_name = 'lambda'
            name = self.expect('IDENT').value
            init_expr = None
            if self.peek() and self.peek().kind == 'OP' and self.peek().value == '=':
                self.i += 1
                init_expr = self.parse_expr()
            if self.peek() and self.peek().value == ';':
                self.i += 1
            return VarDecl(is_const=True, type_name=type_name, name=name, init=init_expr)
        # existing global const array path
        self.i -= 1
        return self.parse_global_const()
    
    def try_parse_main_def(self) -> Optional[FuncDef]:
        is_main = bool(self.match('KW_MAIN'))
        if not is_main:
            return None
            
        if not self.match('KW_DEF'):
            raise SyntaxError("Expected 'def' after '__main__'")
            
        return self.parse_funcdef(is_main)
    
    def try_parse_def(self) -> Optional[FuncDef]:
        if not self.match('KW_DEF'):
            return None
            
        return self.parse_funcdef(False)
    
    def try_parse_extend(self) -> Optional[ASTNode]:
        if not self.match('KW_EXTEND'):
            return None
            
        return self.parse_extend()
    
    def parse_func_decl(self) -> FuncDecl:
        name = self.expect('IDENT').value
        self.expect('SYMBOL')  # '('
        params = self.parse_param_names()
        self.expect('SYMBOL')  # ')'
        
        if self.match('ARROW'):
            while self.peek() and self.peek().value != ';':
                self.i += 1
                
        self.expect('SYMBOL')  # ';'
        return FuncDecl(name=name, params=params, ret="void")
    
    def parse_type_decl(self) -> TypeStruct:
        self.expect('KW_STRUCT')

        # optional: from BaseName
        base_name = None
        if self.match('KW_FROM'):
            base_name = self.expect('IDENT').value

        self.expect('SYMBOL')  # '{'
        fields: List[FieldDecl] = []
        ctor_params: List[str] = []
        
        while self.peek() and self.peek().value != '}':
            is_private = bool(self.match('KW_PRIVATE'))
            t = self.peek()
            
            if t and t.kind == 'IDENT' and t.value == 'ctor':
                self.i += 1
                self.expect('SYMBOL')  # '('
                ctor_params = self.parse_name_list_until(')')
                self.expect('SYMBOL')  # ')'
                self.expect('SYMBOL')  # ';'
                continue
                
            field_type = self.parse_type_name()
            field_name = self.expect('IDENT').value
            default = None
            
            if self.match('OP'):  # '='
                if self.tokens[self.i-1].value != '=':
                    raise SyntaxError("Only '=' allowed in field default")
                default = self.parse_literal_node()
                
            self.expect('SYMBOL')  # ';'
            fields.append(FieldDecl(is_private, field_type, field_name, default))
            
        self.expect('SYMBOL')  # '}'
        name = self.expect('IDENT').value
        self.expect('SYMBOL')  # ';'
        return TypeStruct(name=name, fields=fields, ctor_params=ctor_params, base_name=base_name)
    
    def parse_global_const(self) -> VarDecl:
        t_array = self.expect('IDENT')
        if t_array.value != 'array':
            raise SyntaxError(f"Expected type 'array', found {t_array.value}")
            
        self.expect('SYMBOL')  # '<'
        inner = self.expect('IDENT').value
        self.expect('SYMBOL')  # '>'
        type_name = f"array<{inner}>"
        name = self.expect('IDENT').value

        size_expr = None
        init_expr = None

        # Optional [size]
        if self.peek() and self.peek().value == '[':
            self.i += 1
            size_expr = self.expect('NUMBER').value
            self.expect('SYMBOL')  # ']'

        # Optional initializer
        if self.peek() and self.peek().kind == 'OP' and self.peek().value == '=':
            self.i += 1
            init_expr = self.parse_expr()

        # Eat trailing ';'
        if self.peek() and self.peek().value == ';':
            self.i += 1

        return VarDecl(
            is_const=True,
            type_name=type_name,
            name=name,
            init=init_expr if init_expr is not None else NumberLiteral(int(size_expr)) if size_expr else None
        )
    
    def parse_funcdef(self, is_main: bool) -> FuncDef:
        name = self.expect('IDENT').value
        self.expect('SYMBOL')  # '('
        params = self.parse_param_names()
        self.expect('SYMBOL')  # ')'
        
        if self.match('SCOPE'):
            while self.peek() and self.peek().value != '{':
                self.i += 1
                
        self.expect('SYMBOL')  # '{'
        body = self.parse_block_body()
        self.expect('SYMBOL')  # '}'
        return FuncDef(is_main, name, params, None, body)
    
    def parse_param_names(self) -> List[str]:
        names = []
        while self.peek() and self.peek().value != ')':
            if self.peek().kind == 'IDENT':
                names.append(self.peek().value)
                self.i += 1
            elif self.peek().value == ',':
                self.i += 1
            else:
                self.i += 1
        return names
    
    def parse_block_body(self) -> List[ASTNode]:
        stmts = []
        while self.peek() and self.peek().value != '}':
            t = self.peek()

            # finalize
            if t.kind == 'KW_FINALIZE':
                self.i += 1
                expr = self.parse_expr()
                self._maybe_eat_semicolon()
                stmts.append(Finalize(expr))
                continue

            # const/auto locals
            if t.kind in ('KW_CONST', 'KW_AUTO'):
                stmts.append(self.parse_local_vardecl())
                continue

            # typed locals like: int x = 1;
            if self._looks_like_typed_local():
                stmts.append(self.parse_typed_local_vardecl())
                continue

            # typed-ctor locals like: SomeType x(...);  or  SomeType x();
            if self._looks_like_ctor_decl():
                stmts.append(self.parse_ctor_decl())
                continue

            # this->field = expr;
            if t.kind == 'IDENT' and t.value == 'this':
                self.i += 1
                self.expect('ARROW')
                field = self.expect('IDENT').value
                if self.expect('OP').value != '=':
                    raise SyntaxError("Expected '='")
                expr = self.parse_expr()
                self._maybe_eat_semicolon()
                stmts.append(MemberAssign('this', field, expr))
                continue

            # Everything else goes through parse_statement
            stmt = self.parse_statement()

            # For expression statements only, swallow an optional ';'
            if isinstance(stmt, (CallExpr, VarRef, BinaryExpr, Assign, MemberCall, QualifiedRef, ArrayIndex, ArrayAssignment)):
                self._maybe_eat_semicolon()

            stmts.append(stmt)

        return stmts
    
    def parse_statement(self) -> ASTNode:
        t = self.peek()
        if not t:
            return NumberLiteral(0)
        if t.kind == 'KW_REASSIGN':
                self.i += 1
                target_name = self.expect('IDENT').value
                eq = self.expect('OP')
                if eq.value != '=':
                    raise SyntaxError("Expected '=' after 'reassign NAME'")
                value = self.parse_expr()
                self._maybe_eat_semicolon()
                return Reassign(target=VarRef(target_name), value=value)
        # Handle control flow statements
        if t.kind == 'KW_FOR':
            return self.parse_for()
        if t.kind == 'KW_WHILE':
            return self.parse_while()
        if t.kind == 'KW_IF':
            return self.parse_if()
        if t.kind == 'KW_END':
            self.i += 1
            
            # Check if there's a condition after 'end'
            condition = None
            if self.peek() and self.peek().value != ';':
                condition = self.parse_expr()
                
            self._maybe_eat_semicolon()
            return BreakStmt(condition=condition)
        if t.kind == 'KW_CONT':
            self.i += 1
            self._maybe_eat_semicolon()
            return ContinueStmt()

        # Fallback to an expression statement
        return self.parse_expr()
    
    def parse_for(self) -> ForStmt:
        self.expect('KW_FOR')
        self.expect('SYMBOL')  # '('
        init = self.parse_expr()
        self.expect('SYMBOL')  # ';'
        cond = self.parse_expr()
        self.expect('SYMBOL')  # ';'
        step = self.parse_expr()
        self.expect('SYMBOL')  # ')'
        self.expect('SYMBOL')  # '{'
        body = self.parse_block_body()
        self.expect('SYMBOL')  # '}'
        return ForStmt(init, cond, step, body)
    
    def parse_while(self) -> WhileStmt:
        self.expect('KW_WHILE')
        
        # Parse condition with optional parentheses
        if self.peek() and self.peek().value == '(':
            self.i += 1
            cond = self.parse_expr()
            self.expect('SYMBOL')  # ')'
        else:
            cond = self.parse_expr()
            
        self.expect('SYMBOL')  # '{'
        body = self.parse_block_body()
        self.expect('SYMBOL')  # '}'
        return WhileStmt(cond, body)
    
    def parse_if(self) -> IfStmt:
        self.expect('KW_IF')
        
        # Parse condition with optional parentheses
        if self.peek() and self.peek().value == '(':
            self.i += 1
            cond = self.parse_expr()
            self.expect('SYMBOL')  # ')'
        else:
            cond = self.parse_expr()

        self.expect('SYMBOL')      # '{'
        then_body = self.parse_block_body()
        self.expect('SYMBOL')      # '}'

        else_body = None
        # Handle else and else-if
        if self.peek() and self.peek().kind == 'KW_ELSE':
            self.i += 1
            
            # Check if it's an else-if
            if self.peek() and self.peek().kind == 'KW_IF':
                else_body = [self.parse_if()]  # Recursively parse else-if as nested if
            else:
                self.expect('SYMBOL')  # '{'
                else_body = self.parse_block_body()
                self.expect('SYMBOL')  # '}'

        return IfStmt(cond, then_body, else_body)
    
    def parse_local_vardecl(self) -> VarDecl:
        is_const = bool(self.match('KW_CONST'))
        if not is_const:
            self.expect('KW_AUTO')
            
        type_name = None
        # optional <T> after auto
        if not is_const and self.peek() and self.peek().value == '<':
            self.i += 1
            if self.match('KW_PTR'):
                self.expect('SYMBOL')  # '<'
                inner = self.expect('IDENT').value
                self.expect('SYMBOL')  # '>'
                type_name = f"ptr<{inner}>"
            else:
                type_name = self.expect('IDENT').value
            self.expect('SYMBOL')  # '>'
            
        # name
        name = self.expect('IDENT').value
        # '=' init
        eq = self.expect('OP')
        if eq.value != '=': 
            raise SyntaxError("Expected '=' in var decl")
            
        init = self.parse_expr()
        if self.peek() and self.peek().value == ';':
            self.i += 1
            
        return VarDecl(is_const, type_name, name, init)
    
    def parse_typed_local_vardecl(self) -> VarDecl:
        type_name = self.parse_type_name()
        name = self.expect('IDENT').value
        eq = self.expect('OP')
        if eq.value != '=':
            raise SyntaxError("Expected '=' in typed var decl")
            
        init = self.parse_expr()
        if self.peek() and self.peek().value == ';':
            self.i += 1
            
        return VarDecl(False, type_name, name, init)
    
    def parse_ctor_decl(self) -> VarDecl:
        type_name = self.parse_type_name()
        var_name = self.expect('IDENT').value
        self.expect('SYMBOL')  # '('
        args = self._parse_arg_list()  # consumes ')'
        self._maybe_eat_semicolon()
        return VarDecl(False, type_name, var_name, NewObject(type_name, args))
    
    def parse_expr(self) -> ASTNode:
        return self.parse_assignment()
    
    def parse_assignment(self) -> ASTNode:
        lhs = self.parse_logic()
        
        # Check if it's an array index assignment: array[index] = value
        if isinstance(lhs, ArrayIndex) and self.peek() and self.peek().kind == 'OP' and self.peek().value == '=':
            self.i += 1
            rhs = self.parse_assignment()
            return ArrayAssignment(array=lhs.array, index=lhs.index, value=rhs)
        
        # Regular variable assignment
        if self.peek() and self.peek().kind == 'OP' and self.peek().value == '=':
            if not isinstance(lhs, VarRef):
                raise SyntaxError("Left side of '=' must be a variable or array element")
            self.i += 1
            rhs = self.parse_assignment()
            return Assign(target=lhs, expr=rhs)
        
        return lhs
    
    def parse_logic(self) -> ASTNode:
        node = self.parse_compare()
        while self.peek() and self.peek().kind in ('KW_OR', 'KW_AND'):
            op = self.peek().value
            self.i += 1
            rhs = self.parse_compare()
            node = BinaryExpr(op, node, rhs)
        return node
    
    def parse_compare(self) -> ASTNode:
        node = self.parse_add()
        while True:
            t = self.peek()
            if not t: 
                break
            if t.kind in ('EQEQ','NEQ','LE','GE'):
                self.i += 1
                rhs = self.parse_add()
                node = BinaryExpr(t.kind, node, rhs)
                continue
            if t.kind == 'SYMBOL' and t.value in ('<','>'):
                op = t.value
                self.i += 1
                rhs = self.parse_add()
                node = BinaryExpr(op, node, rhs)
                continue
            break
        return node
    
    def parse_add(self) -> ASTNode:
        node = self.parse_term()
        while self.peek() and self.peek().kind == 'OP' and self.peek().value in ('+','-'):
            op = self.peek().value
            self.i += 1
            rhs = self.parse_term()
            node = BinaryExpr(op, node, rhs)
        return node
    
    def parse_term(self) -> ASTNode:
        node = self.parse_factor()
        while self.peek() and self.peek().kind == 'OP' and self.peek().value in ('*','/'):
            op = self.peek().value
            self.i += 1
            rhs = self.parse_factor()
            node = BinaryExpr(op, node, rhs)
        return node
    
    def parse_factor(self) -> ASTNode:
        t = self.peek()
        if not t:
            raise SyntaxError("Unexpected end of input in expression")
        if t and t.kind == 'OP' and t.value in ('+', '-'):
            self.i += 1
            rhs = self.parse_factor()          # bind tighter than * and +
            if t.value == '+':
                return rhs
            # represent unary minus as (0 - rhs)
            return BinaryExpr('-', NumberLiteral(0), rhs)
        # lambda wrapper literal: [captures](&)(def {...} | funcName)
        if t.value == '[':
            return self._parse_lambda_wrapper()
        if t.kind == 'KW_NIL':
            self.i += 1
            return NilLiteral()
        
        # (addr_of)
        if t.kind == 'AMP':
            self.i += 1
            target = self.parse_factor()
            return CallExpr(
                callee=QualifiedRef(alias='std', name='__addr_of__'),
                args=[target]
            )

        # (expr)
        if t.value == '(':
            self.i += 1
            node = self.parse_expr()
            self.expect('SYMBOL')  # ')'
            return self._parse_postfix(node)

        # NUMBER
        if t.kind == 'NUMBER':
            self.i += 1
            node = NumberLiteral(float(t.value) if '.' in t.value else int(t.value))
            return self._parse_postfix(node)

        # STRING
        if t.kind == 'STRING':
            self.i += 1
            s_clean = t.value.strip('"').strip('“').strip('”')
            node = StringLiteral(s_clean)
            return self._parse_postfix(node)

        # NEW: delegate collection constructor new[](&)( defaultOrNull? )
        if t.kind in ('KW_NEW','IDENT') and t.value == 'new':
            self.i += 1  # consume 'new'
            if self.peek() and self.peek().value == '[':
                self.i += 1
                # distinguish new[](&) from new[ size ]
                if self.peek() and self.peek().value == ']':
                    self.i += 1  # ']'
                    # expect (&)
                    self.expect('SYMBOL')  # '('
                    if not (self.peek() and self.peek().kind == 'AMP'):
                        raise SyntaxError("Expected '&' inside new[](&)")
                    self.i += 1  # '&'
                    self.expect('SYMBOL')  # ')'
                    # now ( default? )
                    self.expect('SYMBOL')  # '('
                    default_cb = None
                    if self.peek() and self.peek().value != ')':
                        if self.peek().value == '[':
                            default_cb = self._parse_lambda_wrapper()
                        else:
                            # allow passing a named wrapper or null/func
                            default_cb = self.parse_expr()
                        # default_cb must evaluate to wrapper or None
                    self.expect('SYMBOL')  # ')'
                    return self._parse_postfix(NewDelegateCollection(bound=True, default=default_cb))
                else:
                    # classic array new[ size ]
                    size_expr = self.parse_expr()
                    self.expect('SYMBOL')  # ']'
                    return NewArray(size=size_expr)

            # fallback: new TypeName(args)
            type_tok = self.expect('IDENT')
            self.expect('SYMBOL')
            args = self._parse_arg_list()
            return self._parse_postfix(NewObject(type_name=type_tok.value, args=args))

        # IDENT (variable)
        if t.kind == 'IDENT':
            self.i += 1
            if self.peek() and self.peek().kind == 'SCOPE':  # ':'
                self.i += 1
                name = self.expect('IDENT').value
                node = QualifiedRef(alias=t.value, name=name)
                return self._parse_postfix(node)
            node = VarRef(t.value)
            return self._parse_postfix(node)

        # If we reach here, it's an unexpected token
        raise SyntaxError(f"Unexpected token in expression: {t}")
    
    def _parse_lambda_wrapper(self) -> LambdaWrapper:
        # [capture0, capture1, ...]
        self.expect('SYMBOL')  # '['
        captures: List[ASTNode] = []
        while self.peek() and self.peek().value != ']':
            captures.append(self.parse_expr())
            if self.peek() and self.peek().value == ',':
                self.i += 1
        self.expect('SYMBOL')  # ']'

        # (&) required per your syntax
        self.expect('SYMBOL')  # '('
        if not (self.peek() and self.peek().kind == 'AMP'):
            raise SyntaxError("Expected '&' inside '(&)' of lambda wrapper")
        self.i += 1  # consume '&'
        self.expect('SYMBOL')  # ')'

        # (def {...} | def -> T {...} | funcName)
        self.expect('SYMBOL')  # '('
        func_node: Union[LambdaLiteral, VarRef]
        if self.peek() and self.peek().kind == 'KW_DEF':
            self.i += 1  # 'def'
            ret = None
            if self.match('ARROW'):
                # parse return type name tokens up to '{'
                ret = []
                while self.peek() and self.peek().value != '{':
                    if self.peek().kind in ('IDENT',):
                        ret.append(self.peek().value)
                    self.i += 1
                ret = ' '.join(ret) if ret else None
            self.expect('SYMBOL')  # '{'
            body = self.parse_block_body()
            self.expect('SYMBOL')  # '}'
            func_node = LambdaLiteral(params=[], ret=ret, body=body)
        else:
            # a reference to a declared function
            name = self.expect('IDENT').value
            func_node = VarRef(name)

        self.expect('SYMBOL')  # ')'
        return LambdaWrapper(captures=captures, bound=True, func=func_node)

    def _parse_postfix(self, node: ASTNode) -> ASTNode:
        while True:
            if self.peek() and self.peek().value == '[':
                self.i += 1
                index = self.parse_expr()
                self.expect('SYMBOL')  # ']'
                node = ArrayIndex(array=node, index=index)
                continue

            if self.peek() and self.peek().kind == 'ARROW':
                self.i += 1
                member = self.expect('IDENT').value
                # method call: ->name(...)
                if self.peek() and self.peek().value == '(':
                    self.i += 1
                    args = []
                    while self.peek() and self.peek().value != ')':
                        args.append(self.parse_expr())
                        if self.peek() and self.peek().value == ',':
                            self.i += 1
                    self.expect('SYMBOL')  # ')'
                    node = MemberCall(receiver=node, member=member, args=args)
                    continue
                # field get: ->name
                node = FieldGet(receiver=node, field=member)
                continue

            if self.peek() and self.peek().value == '(':
                self.i += 1
                args = []
                while self.peek() and self.peek().value != ')':
                    args.append(self.parse_expr())
                    if self.peek() and self.peek().value == ',':
                        self.i += 1
                self.expect('SYMBOL')  # ')'
                node = CallExpr(callee=node, args=args)
                continue

            if (self.peek() and 
                (self.peek().value in (';', '}', ')', ']', ','))):
                break

            break
        return node

    
    def parse_extend(self) -> Optional[ASTNode]:
        if self.match('KW_DEF'):
            name = self.expect('IDENT').value
            self.expect('SYMBOL')  # '('
            _ = self.parse_param_names()
            self.expect('SYMBOL')  # ')'
            if self.match('ARROW'):
                while self.peek() and self.peek().value != '{':
                    self.i += 1
            self.expect('SYMBOL')  # '{'
            body = self.parse_block_body()
            self.expect('SYMBOL')  # '}'
            return ExtendFunc(name=name, params=[], ret=None, body=body)

        if self.match('KW_TYPE'):
            type_name = self.expect('IDENT').value
            self.expect('SCOPE')  # ':'
            ident = self.expect('IDENT').value

            # ---- ctor extension (existing) ----
            if ident == 'ctor':
                self.expect('SYMBOL')  # '('
                _params = self.parse_param_names()
                self.expect('SYMBOL')  # ')'

                base_delegate = None
                if self.match('KW_INTO'):
                    base_ident = self.expect('IDENT').value
                    if base_ident != 'base':
                        raise SyntaxError("Only 'into base(...)' supported")
                    self.expect('SYMBOL')  # '('
                    base_args = []
                    while self.peek() and self.peek().value != ')':
                        base_args.append(self.parse_expr())
                        if self.peek() and self.peek().value == ',':
                            self.i += 1
                    self.expect('SYMBOL')  # ')'
                    base_delegate = ('base', base_args)

                self.expect('SYMBOL')  # '{'
                body = self.parse_block_body()
                self.expect('SYMBOL')  # '}'
                return ExtendCtor(type_name=type_name, params=_params, body=body, base_delegate=base_delegate)

            # ---- instance method extension ----
            method_name = ident
            self.expect('SYMBOL')  # '('
            params = self.parse_param_names()
            self.expect('SYMBOL')  # ')'

            ret = None
            if self.match('ARROW'):
                # slurp simple return type tokens until '{'
                parts = []
                while self.peek() and self.peek().value != '{':
                    if self.peek().kind in ('IDENT',):
                        parts.append(self.peek().value)
                    self.i += 1
                ret = ' '.join(parts) if parts else None

            self.expect('SYMBOL')  # '{'
            body = self.parse_block_body()
            self.expect('SYMBOL')  # '}'
            return ExtendMethod(type_name=type_name, method=method_name, params=params, ret=ret, body=body)

        # no match
        self.i += 1
        return None

    
    def parse_literal_node(self) -> ASTNode:
        t = self.peek()
        if not t:
            raise SyntaxError("Expected literal")

        # support negative numeric literal in defaults
        if t.kind == 'OP' and t.value == '-':
            self.i += 1
            num = self.expect('NUMBER').value
            return NumberLiteral(-float(num) if '.' in num else -int(num))

        if t.kind == 'NUMBER':
            self.i += 1
            return NumberLiteral(float(t.value) if '.' in t.value else int(t.value))

        if t.kind == 'STRING':
            self.i += 1
            return StringLiteral(t.value.strip('"').strip('“').strip('”'))

        raise SyntaxError(f"Expected number or string, found {t}")

    def parse_type_name(self) -> str:
        parts = []
        if self.match('KW_CONST'):
            parts.append('const ')
            
        if self.match('KW_PTR'):
            parts.append('ptr')
            self.expect('SYMBOL')  # '<'
            inner = self.expect('IDENT').value
            self.expect('SYMBOL')  # '>'
            return ''.join(parts) + f"<{inner}>"
            
        tok = self.expect('IDENT')
        base = tok.value
        if self.peek() and self.peek().value == '<':
            self.i += 1  # '<'
            inner = self.expect('IDENT').value
            self.expect('SYMBOL')  # '>'
            return ''.join(parts) + f"{base}<{inner}>"
            
        return ''.join(parts) + base
    
    def parse_name_list_until(self, closing: str) -> List[str]:
        names = []
        while self.peek() and self.peek().value != closing:
            if self.peek().kind == 'IDENT':
                names.append(self.peek().value)
                self.i += 1
            else:
                self.i += 1
        return names
    
    def _parse_arg_list(self) -> List[ASTNode]:
        args = []
        while self.peek() and self.peek().value != ')':
            args.append(self.parse_expr())
            if self.peek() and self.peek().value == ',':
                self.i += 1
        self.expect('SYMBOL') # ')'
        return args
    
    def _maybe_eat_semicolon(self):
        if self.peek() and self.peek().value == ';':
            self.i += 1
    
    def _look(self, k: int) -> Optional[Token]:
        j = self.i + k
        return self.tokens[j] if j < len(self.tokens) else None
    
    def _looks_like_typed_local(self) -> bool:
        t0 = self._look(0)  # type
        t1 = self._look(1)  # name
        t2 = self._look(2)  # '=' or '('
        return (t0 and t0.kind == 'IDENT' and
                t1 and t1.kind == 'IDENT' and
                t2 and t2.kind == 'OP' and t2.value == '=')
    
    def _looks_like_ctor_decl(self) -> bool:
        t0 = self._look(0)  # type
        t1 = self._look(1)  # name
        t2 = self._look(2)  # '(' or '='
        return (t0 and t0.kind == 'IDENT' and
                t1 and t1.kind == 'IDENT' and
                t2 and t2.value == '(')
    
    # Helper methods
    def peek(self) -> Optional[Token]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def match(self, kind: str) -> Optional[Token]:
        t = self.peek()
        if t and t.kind == kind:
            self.i += 1
            return t
        return None

    def expect(self, kind: str) -> Token:
        t = self.peek()
        if not t or t.kind != kind:
            raise SyntaxError(f"Expected {kind} but found {t}")
        self.i += 1
        return t

# ==================== SEMANTIC ANALYSIS ====================

class SemanticAnalyzer(CompilerComponent):
    """Performs semantic analysis on the AST."""
    
    def __init__(self, compiler: 'Compiler'):
        super().__init__(compiler)
        self.types: Set[str] = set()
        self.funcs: Set[str] = set()
        self.modules: Dict[str, ModuleSymbols] = {}
        self.type_methods: Dict[str, Set[str]] = {}
        self.inherits: Dict[str, str] = {}
        self._loop_depth = 0
        self._function_depth = 0
    
    def analyze(self, program: Program):
        """Perform semantic analysis on the program."""
        self.check_program(program)
    
    def check_program(self, prog: Program):
        # Check imports
        for item in prog.items:
            if isinstance(item, ImportStmt):
                if item.alias and item.alias not in self.modules:
                    raise SemanticError(
                        f"Imported alias '{item.alias}' is not loaded. "
                        f"Load its file on the CLI."
                    )

        
        # Collect declarations
        for item in prog.items:
            self.register_decl(item)
        
        # Check items
        for item in prog.items:
            if isinstance(item, VarDecl):
                self.check_vardecl(item)
            elif isinstance(item, FuncDef):
                self.check_funcdef(item)
            elif isinstance(item, (ExtendFunc, ExtendCtor)):
                pass  # Not fully enforced yet
            elif isinstance(item, ExtendCtor):
                if item.base_delegate and item.type_name not in self.inherits:
                    raise SemanticError(f"'{item.type_name}' has no base to delegate into")
        
        # Validate inheritance
        for child, base in self.inherits.items():
            if base not in self.types:
                raise SemanticError(f"Unknown base type '{base}' for '{child}'")
        
        # Check for cycles
        self.check_inheritance_cycles()
    
    def check_inheritance_cycles(self):
        """Check for inheritance cycles."""
        seen = set()
        
        def _walk(t):
            while t in self.inherits:
                if t in seen:
                    return True
                seen.add(t)
                t = self.inherits[t]
            return False
        
        for t in list(self.inherits.keys()):
            seen.clear()
            if _walk(t):
                raise SemanticError("Inheritance cycle detected")
    
    def register_decl(self, item: ASTNode):
        if isinstance(item, ExtendMethod):
            self.type_methods.setdefault(item.type_name, set()).add(item.method)

        if isinstance(item, FuncDecl):
            self.funcs.add(item.name)
        if isinstance(item, TypeStruct):
            self.types.add(item.name)
            if getattr(item, "base_name", None):
                self.inherits[item.name] = item.base_name
    
    def check_stmt(self, stmt: ASTNode):
        if isinstance(stmt, (BreakStmt, ContinueStmt)):
            if self._loop_depth == 0:
                raise SemanticError(("'end'" if isinstance(stmt, BreakStmt) else "'cont'") + " used outside of loop")
            return
            
        if isinstance(stmt, VarDecl):
            self.check_vardecl(stmt)
            return
            
        if isinstance(stmt, Finalize):
            self.check_expr(stmt.expr)
            return
            
        if isinstance(stmt, MemberAssign):
            self.check_expr(stmt.expr)
            return
            
        if isinstance(stmt, IfStmt):
            self.check_expr(stmt.cond)
            for x in stmt.then_body: 
                self.check_stmt(x)
            if stmt.else_body:
                for x in stmt.else_body: 
                    self.check_stmt(x)
            return
            
        if isinstance(stmt, WhileStmt):
            self.check_expr(stmt.cond)
            self._loop_depth += 1
            for x in stmt.body: 
                self.check_stmt(x)
            self._loop_depth -= 1
            return
            
        if isinstance(stmt, ForStmt):
            self.check_expr(stmt.init)
            self.check_expr(stmt.cond)
            self.check_expr(stmt.step)
            self._loop_depth += 1
            for x in stmt.body: 
                self.check_stmt(x)
            self._loop_depth -= 1
            return

        # Expression statements (CallExpr, etc.)
        if isinstance(stmt, Assign):
            self.check_expr(stmt.expr)
            return
        
        if isinstance(stmt, Reassign):
            self.check_expr(stmt.value)
            return

        # Fallback: treat as expression statement
        self.check_expr(stmt)
    
    def check_vardecl(self, vd: VarDecl):
        self.check_type_exists(vd.type_name)
        self.check_expr(vd.init)
    
    def check_funcdef(self, f: FuncDef):
        self._function_depth += 1
        for stmt in f.body:
            self.check_stmt(stmt)
        self._function_depth -= 1
    
    def check_type_exists(self, type_name: Optional[str]):
        if not type_name:
            return
            
        base = type_name.split('<', 1)[0].strip()
        if base not in self.types:
            raise SemanticError(f"Unknown type '{base}' used")
    
    def check_expr(self, e: ASTNode):
        if isinstance(e, (NumberLiteral, StringLiteral, VarRef, ArrayIndex, NewArray, NilLiteral)):
            return
        if isinstance(e, FieldGet):
            self.check_expr(e.receiver)
            return
        if isinstance(e, NewObject):
            if e.type_name not in self.types:
                raise SemanticError(f"Unknown type '{e.type_name}'")
            for a in e.args:
                self.check_expr(a)
            return
            
        if isinstance(e, ArrayAssignment):
            self.check_expr(e.array)
            self.check_expr(e.index)
            self.check_expr(e.value)
            return
            
        if isinstance(e, BinaryExpr):
            self.check_expr(e.left)
            self.check_expr(e.right)
            return
            
        if isinstance(e, QualifiedRef):
            raise SemanticError("Qualified name used where a value is required")
            
        if isinstance(e, MemberCall):
            self.check_expr(e.receiver)
            for a in e.args:
                self.check_expr(a)
            return
            
        if isinstance(e, LambdaWrapper):
            for c in e.captures:
                self.check_expr(c)
            # if VarRef func, ensure callable name exists if declared
            if isinstance(e.func, VarRef) and (e.func.name not in self.funcs):
                # allow late-binding: do not hard fail, but you can enforce if you want:
                # raise SemanticError(f"Unknown function '{e.func.name}' in lambda wrapper")
                pass
            if isinstance(e.func, LambdaLiteral):
                self.check_expr(e.func)
            return

        if isinstance(e, NewDelegateCollection):
            if e.default:
                self.check_expr(e.default)
            return

        if isinstance(e, CallExpr):
            cal = e.callee
            if isinstance(cal, VarRef):
                if cal.name not in self.funcs:
                    raise SemanticError(f"Call to undeclared function '{cal.name}'")
            elif isinstance(cal, QualifiedRef):
                alias = cal.alias
                name = cal.name
                mod = self.modules.get(alias)
                if not mod or name not in mod.funcs:
                    raise SemanticError(f"Call to undeclared function '{alias}:{name}'")
            else:
                raise SemanticError("Unsupported callee expression")
                
            for a in e.args:
                self.check_expr(a)
            return
            
        if isinstance(e, LambdaLiteral):
            for s in e.body:
                self.check_stmt(s)
            return
        
        if isinstance(e, Assign): 
            self.check_expr(e.expr)
            return
        
        if isinstance(e, Reassign):
            self.check_expr(e.value)
            return
        
        raise SemanticError(f"Unsupported expression: {type(e).__name__}")

# ==================== EXECUTION ====================

class Executor(CompilerComponent):
    """Executes the program."""
    
    def __init__(self, compiler: 'Compiler'):
        super().__init__(compiler)
        self.program: Optional[Program] = None
        self.globals: Dict[str, Any] = {}
        self.main: Optional[FuncDef] = None
        self.runtime = self.create_runtime()
        self.types: Dict[str, TypeStruct] = {}
        self.ext_ctors: Dict[str, ExtendCtor] = {}
        self.this_stack: List[Dict[str, Any]] = []
        self.user_funcs: Dict[str, FuncDef] = {}
        self.inherits: Dict[str, str] = {}
        self.type_methods: Dict[str, Dict[str, ExtendMethod]] = {}
    
    def create_runtime(self) -> Dict:
        try:
            from runtime import createRuntimeStd
        except ImportError:
            from .runtime import createRuntimeStd
            
        return createRuntimeStd()
    
    def execute(self, program: Program):
        """Execute the program."""
        self.program = program
        self.initialize()
        
        if not self.main:
            print("No __main__ function found.")
            return
        
        frame = {p: 0 for p in self.main.params}
        try:
            for stmt in self.main.body:
                self.exec_stmt(stmt, frame)
        except _ReturnSignal as rs:
            if rs.value is not None:
                print(rs.value)
    
    def initialize(self):
        """Initialize the execution environment."""
        for item in self.program.items:
            if isinstance(item, TypeStruct):
                self.types[item.name] = item
            elif isinstance(item, ExtendCtor):
                self.ext_ctors[item.type_name] = item
            elif isinstance(item, ExtendMethod):
                self.type_methods.setdefault(item.type_name, {})[item.method] = item
            elif isinstance(item, VarDecl):
                val = self.eval_expr(item.init, {})
                self.globals[item.name] = val
            elif isinstance(item, FuncDef):
                self.user_funcs[item.name] = item
                if item.is_main:
                    self.main = item
    
    def exec_stmt(self, stmt: ASTNode, frame: Dict[str, Any]):
        if isinstance(stmt, VarDecl):
            frame[stmt.name] = self.eval_expr(stmt.init, frame)
            return
        if isinstance(stmt, Reassign):
                val = self.eval_expr(stmt.value, frame)
                name = stmt.target.name if isinstance(stmt.target, VarRef) else str(stmt.target)
                self.globals[name] = val
                return
        
        if isinstance(stmt, Finalize):
            val = self.eval_expr(stmt.expr, frame)
            raise _ReturnSignal(val)

        if isinstance(stmt, IfStmt):
            cond_val = self.eval_expr(stmt.cond, frame)
            if cond_val:
                for s in stmt.then_body:
                    self.exec_stmt(s, frame)
            elif stmt.else_body:
                for s in stmt.else_body:
                    self.exec_stmt(s, frame)
            return

        if isinstance(stmt, ForStmt):
            self.eval_expr(stmt.init, frame)
            try:
                while self.eval_expr(stmt.cond, frame):
                    try:
                        for s in stmt.body:
                            self.exec_stmt(s, frame)
                    except _ContinueSignal:
                        pass
                    self.eval_expr(stmt.step, frame)
            except _BreakSignal:
                pass
            return

        if isinstance(stmt, WhileStmt):
            try:
                while self.eval_expr(stmt.cond, frame):
                    try:
                        for s in stmt.body:
                            self.exec_stmt(s, frame)
                    except _ContinueSignal:
                        pass
            except _BreakSignal:
                pass
            return

        if isinstance(stmt, BreakStmt):
            if stmt.condition:
                condition_met = self.eval_expr(stmt.condition, frame)
                if condition_met:
                    raise _BreakSignal()
            else:
                raise _BreakSignal()
            return

        if isinstance(stmt, ContinueStmt):
            raise _ContinueSignal()
        
        if isinstance(stmt, MemberAssign):
            if stmt.target_object == 'this':
                if not self.this_stack:
                    return
                cur = self.this_stack[-1]
                val = self.eval_expr(stmt.expr, frame)
                cur[stmt.field] = val
                return

        # expression or call
        self.eval_expr(stmt, frame)
    
    def eval_expr(self, expr: ASTNode, locals_frame: Dict[str, Any]) -> Any:
        if isinstance(expr, NilLiteral):
            return None
        if isinstance(expr, VarRef):
            return locals_frame.get(expr.name, self.globals.get(expr.name))
        if isinstance(expr, StringLiteral):
            return expr.value
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, NewArray):
            size = self.eval_expr(expr.size, locals_frame)
            return [0] * int(size)
        if isinstance(expr, ArrayIndex):
            array = self.eval_expr(expr.array, locals_frame)
            index = self.eval_expr(expr.index, locals_frame)
            if isinstance(array, (list, str)) and isinstance(index, int) and 0 <= index < len(array):
                return array[index]
            return None
        if isinstance(expr, ArrayAssignment):
            array = self.eval_expr(expr.array, locals_frame)
            index = self.eval_expr(expr.index, locals_frame)
            value = self.eval_expr(expr.value, locals_frame)
            if isinstance(array, list) and 0 <= index < len(array):
                array[index] = value
            return value
        if isinstance(expr, BinaryExpr):
            l = self.eval_expr(expr.left, locals_frame)
            r = self.eval_expr(expr.right, locals_frame)
            if expr.op == '+':  return (l or 0) + (r or 0)
            if expr.op == '-':  return (l or 0) - (r or 0)
            if expr.op == '*':  return (l or 0) * (r or 0)
            if expr.op == '/':  return (l or 0) / (r or 1)
            if expr.op == '<':  return (l or 0) < (r or 0)
            if expr.op == '>':  return (l or 0) > (r or 0)
            if expr.op == 'LE' or expr.op == '<=':  return (l or 0) <= (r or 0)
            if expr.op == 'GE' or expr.op == '>=':  return (l or 0) >= (r or 0)
            if expr.op == 'EQEQ' or expr.op == '==': return l == r
            if expr.op == 'NEQ' or expr.op == '!=':  return l != r
            if expr.op == 'or': return bool(l) or bool(r)
            if expr.op == 'and': return bool(l) and bool(r)
            return None
        if isinstance(expr, Assign):
            val = self.eval_expr(expr.expr, locals_frame)
            name = expr.target.name
            if name in locals_frame:
                locals_frame[name] = val
            elif name in self.globals:
                self.globals[name] = val
            else:
                locals_frame[name] = val
            return val
        if isinstance(expr, CallExpr):
            cal = expr.callee
            args = [self.eval_expr(a, locals_frame) for a in expr.args]

            # User-defined functions: VarRef
            if isinstance(cal, VarRef) and cal.name in self.user_funcs:
                fdef = self.user_funcs[cal.name]
                new_frame = {p: (args[i] if i < len(args) else None) for i, p in enumerate(fdef.params)}
                try:
                    for s in fdef.body:
                        self.exec_stmt(s, new_frame)
                except _ReturnSignal as rs:
                    return rs.value
                return None

            # User-defined functions: QualifiedRef (e.g. std:push)
            if isinstance(cal, QualifiedRef) and cal.name in self.user_funcs:
                fdef = self.user_funcs[cal.name]
                new_frame = {p: (args[i] if i < len(args) else None) for i, p in enumerate(fdef.params)}
                try:
                    for s in fdef.body:
                        self.exec_stmt(s, new_frame)
                except _ReturnSignal as rs:
                    return rs.value
                return None

            # Built-in runtime
            if isinstance(cal, QualifiedRef):
                fn = self.runtime.get((cal.alias, cal.name))
                if fn:
                    return fn(*args)

            return None
        if isinstance(expr, LambdaLiteral):
                # produce a callable package; we keep AST to execute later
                return {"__kind": "lambda_literal", "node": expr}

        if isinstance(expr, LambdaWrapper):
            # evaluate captures now (at add-time semantics)
            cap_vals = [self.eval_expr(c, locals_frame) for c in expr.captures]
            func_desc = None
            if isinstance(expr.func, VarRef):
                func_desc = ("funcdef", expr.func.name)
            else:
                func_desc = ("lambda", {"__kind": "lambda_literal", "node": expr.func})
            return {"__kind": "lambda_wrapper",
                    "captures": cap_vals,
                    "bound": bool(expr.bound),
                    "func": func_desc}

        if isinstance(expr, NewDelegateCollection):
            coll = {"__kind": "delegate_collection", "handlers": []}
            if expr.default:
                dv = self.eval_expr(expr.default, locals_frame)
                if isinstance(dv, dict) and dv.get("__kind") == "lambda_wrapper":
                    coll["handlers"].append(dv)
                elif dv is None:
                    pass
                else:
                    # allow passing VarRef to function as default
                    # normalize to wrapper with empty captures
                    if isinstance(expr.default, VarRef):
                        coll["handlers"].append({"__kind":"lambda_wrapper",
                                                "captures": [],
                                                "bound": True,
                                                "func": ("funcdef", expr.default.name)})
            return coll
        if isinstance(expr, FieldGet):
            recv = self.eval_expr(expr.receiver, locals_frame)
            if isinstance(recv, dict):
                return recv.get(expr.field)
            return None
        
        if isinstance(expr, MemberCall):
            recv = self.eval_expr(expr.receiver, locals_frame)
            arg_vals = [self.eval_expr(a, locals_frame) for a in expr.args]

            if isinstance(recv, dict) and recv.get("__kind") == "delegate_collection":
                m = expr.member

                if m == "add":
                    w = arg_vals[0] if arg_vals else None
                    if isinstance(w, dict) and w.get("__kind") == "lambda_wrapper":
                        recv["handlers"].append(w)
                    return None
                if m == "remove":
                    w = arg_vals[0] if arg_vals else None
                    if isinstance(w, dict) and w.get("__kind") == "lambda_wrapper":
                        recv["handlers"] = [x for x in recv["handlers"] if x is not w]
                    elif isinstance(w, str):
                        recv["handlers"] = [x for x in recv["handlers"]
                                            if not (x.get("func") == ("funcdef", w))]
                    return None
                if m == "call":
                    # flatten call-time args (keep as list)
                    call_args = arg_vals
                    last = None
                    for w in list(recv["handlers"]):
                        res = self._invoke_wrapper(w, call_args)
                        if res is not None:
                            last = res
                    return last
                # unknown method -> ignore
                return None

            if isinstance(recv, dict) and "__type" in recv:
                return self._call_method(recv, expr.member, arg_vals)
                
            # ordinary member calls not supported elsewhere
            return None
            
        if isinstance(expr, NewObject):
            args = [self.eval_expr(a, locals_frame) for a in expr.args]
            return self._instantiate(expr.type_name, args)

        return None
    def _call_method(self, obj: Dict[str, Any], name: str, args: List[Any]) -> Any:
        t = obj.get("__type")
        while t:
            mtbl = self.type_methods.get(t)
            if mtbl and name in mtbl:
                m: ExtendMethod = mtbl[name]
                frame = {p: (args[i] if i < len(args) else None)
                        for i, p in enumerate(m.params)}
                frame["this"] = obj                     # <-- add this
                self.this_stack.append(obj)
                try:
                    for s in m.body:
                        self.exec_stmt(s, frame)
                except _ReturnSignal as rs:
                    return rs.value
                finally:
                    self.this_stack.pop()
                return None
            T = self.types.get(t)
            t = T.base_name if T else None
        return None

    def _build_object_skel(self, type_name: str) -> Dict[str, Any]:
        """Create an object dict with fields (including inherited) initialized to defaults."""
        chain = []
        t = self.types.get(type_name)
        while t:
            chain.append(t)
            if not t.base_name: 
                break
            t = self.types.get(t.base_name)
        chain = list(reversed(chain))

        obj = {"__type": type_name, "__base": None}
        for t in chain:
            for fld in t.fields:
                if fld.default is not None:
                    obj[fld.name] = self.eval_expr(fld.default, {})
                else:
                    obj[fld.name] = 0
                    
        if len(chain) > 1:
            base_type = chain[-2].name
            obj["__base"] = {"__type": base_type}
            
        return obj
    
    def _invoke_wrapper(self, w: Dict[str, Any], call_args: List[Any]) -> Any:
        if not (isinstance(w, dict) and w.get("__kind") == "lambda_wrapper"):
            return None
        captures = w.get("captures", [])
        effective_args = list(captures) + list(call_args)

        fkind, fval = w.get("func", (None, None))
        if fkind == "funcdef":
            return self._call_user_func_by_name(fval, effective_args)
        if fkind == "lambda":
            lit = fval.get("node")
            return self._run_lambda_literal(lit, effective_args)
        return None

    def _call_user_func_by_name(self, name: str, args: List[Any]) -> Any:
        fdef = self.user_funcs.get(name)
        if not fdef:
            return None
        new_frame = {p: (args[i] if i < len(args) else None)
                    for i, p in enumerate(fdef.params)}
        try:
            for s in fdef.body:
                self.exec_stmt(s, new_frame)
        except _ReturnSignal as rs:
            return rs.value
        return None

    def _run_lambda_literal(self, lit: LambdaLiteral, args: List[Any]) -> Any:
        # expose args list to the lambda body
        frame = {"args": args}
        # expose 'this' if bound context exists (best-effort)
        if self.this_stack:
            frame["this"] = self.this_stack[-1]
        try:
            for s in lit.body:
                self.exec_stmt(s, frame)
        except _ReturnSignal as rs:
            return rs.value
        return None

    def _call_ctor_for_type(self, type_name: str, obj: Dict[str, Any], args: List[Any]):
        """Find and run a ctor for this type or nearest base that has one."""
        t = type_name
        while t:
            ctor = self.ext_ctors.get(t)
            if ctor:
                frame = {}
                for i, p in enumerate(ctor.params):
                    frame[p] = args[i] if i < len(args) else None
                # base delegation unchanged...
                self.this_stack.append(obj)
                frame["this"] = obj                              # <-- add this
                try:
                    for s in ctor.body:
                        self.exec_stmt(s, frame)
                finally:
                    self.this_stack.pop()
                return

            ti = self.types.get(t)
            t = ti.base_name if ti else None

    def _instantiate(self, type_name: str, args: List[Any]) -> Dict[str, Any]:
        obj = self._build_object_skel(type_name)
        self._call_ctor_for_type(type_name, obj, args)
        return obj

# ==================== EXTENSION SYSTEM ====================

class CompilerExtension(ABC):
    """Base class for compiler extensions."""
    
    def __init__(self, compiler: 'Compiler'):
        self.compiler = compiler
    
    @abstractmethod
    def apply(self):
        """Apply this extension to the compiler."""
        pass
    
    @abstractmethod
    def remove(self):
        """Remove this extension from the compiler."""
        pass

# ==================== COMPILER MAIN CLASS ====================

class Compiler:
    """Main compiler class that coordinates all components."""
    
    def __init__(self):
        self.tokenizer = Tokenizer(self)
        self.parser = Parser(self)
        self.semantic_analyzer = SemanticAnalyzer(self)
        self.executor = Executor(self)
        
        # Register extensions
        self.extensions: Dict[str, CompilerExtension] = {}
    
    def register_extension(self, name: str, extension: Type[CompilerExtension]):
        """Register a compiler extension."""
        self.extensions[name] = extension(self)
    
    def enable_extension(self, name: str):
        """Enable a compiler extension."""
        if name in self.extensions:
            self.extensions[name].apply()
    
    def disable_extension(self, name: str):
        """Disable a compiler extension."""
        if name in self.extensions:
            self.extensions[name].remove()
    
    def compile(self, source: str) -> Program:
        """Compile source code to an AST."""
        tokens = self.tokenizer.tokenize(source)
        ast = self.parser.parse(tokens)
        self.semantic_analyzer.analyze(ast)
        return ast
    
    def execute(self, program: Program):
        """Execute a program."""
        self.executor.execute(program)
    
    def compile_and_execute(self, source: str):
        """Compile and execute source code."""
        program = self.compile(source)
        self.execute(program)

# ==================== MODULE LOADING ====================

@dataclass
class ModuleBundle:
    name: str
    path: str
    program: Program

def _read_sources(paths: List[str]) -> List[Tuple[str, str]]:
    """Return list of (path, text)."""
    out = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out.append((os.path.abspath(p), f.read()))
    return out

def _modname_from_path(path: str) -> str:
    """Module name is filename stem: /x/y/std.nl -> 'std'."""
    return os.path.splitext(os.path.basename(path))[0]

def _parse_modules(sources: List[Tuple[str, str]], compiler: Compiler) -> List[ModuleBundle]:
    bundles: List[ModuleBundle] = []
    for path, src in sources:
        toks = compiler.tokenizer.tokenize(src)
        prog = compiler.parser.parse(toks)
        bundles.append(ModuleBundle(_modname_from_path(path), path, prog))
    return bundles

def _collect_module_symbols(analyzer: SemanticAnalyzer, bundles: List[ModuleBundle]):
    """First pass: populate analyzer.modules with exported symbols per module."""
    analyzer.modules = {}
    for b in bundles:
        mod = ModuleSymbols(types=set(), funcs=set())
        for it in b.program.items:
            if isinstance(it, FuncDecl):
                mod.funcs.add(it.name)
            if isinstance(it, TypeStruct):
                mod.types.add(it.name)
        analyzer.modules[b.name] = mod
def _abs_from(base_path: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(os.path.dirname(base_path), rel))

def _load_imports_recursively(bundles: List[ModuleBundle], compiler: Compiler) -> List[ModuleBundle]:
    seen = {os.path.abspath(b.path) for b in bundles}
    i = 0
    while i < len(bundles):
        b = bundles[i]
        for it in getattr(b.program, "items", []):
            if isinstance(it, ImportStmt):
                imp_path = _abs_from(b.path, it.path)
                if imp_path not in seen:
                    with open(imp_path, "r", encoding="utf-8") as f:
                        src = f.read()
                    toks = compiler.tokenizer.tokenize(src)
                    prog = compiler.parser.parse(toks)
                    bundles.append(ModuleBundle(_modname_from_path(imp_path), imp_path, prog))
                    seen.add(imp_path)
        i += 1
    return bundles

def _resolve_imports(analyzer: SemanticAnalyzer, bundles: List[ModuleBundle]):
    # Build/rebuild modules first
    _collect_module_symbols(analyzer, bundles)

    # Path → module name for alias mapping
    path_to_modname = {os.path.abspath(b.path): _modname_from_path(b.path) for b in bundles}

    # Wire aliases (if any). If alias-less, nothing to wire—types/methods are already loaded.
    for b in bundles:
        for it in getattr(b.program, "items", []):
            if isinstance(it, ImportStmt) and it.alias:
                imp_abs = _abs_from(b.path, it.path)
                modname = path_to_modname.get(imp_abs)
                if modname and modname in analyzer.modules:
                    analyzer.modules[it.alias] = analyzer.modules[modname]
                else:
                    # If the file didn’t load for some reason:
                    raise SemanticError(f"Could not resolve import '{it.path}' for alias '{it.alias}'.")


def _merge_programs(bundles: List[ModuleBundle]) -> Program:
    """Create a single Program by concatenating items in CLI order."""
    merged_items = []
    for b in bundles:
        merged_items.extend(b.program.items)
    return Program(merged_items)

def _resolve_path(from_path: str, import_path: str) -> str:
    if os.path.isabs(import_path):
        return os.path.abspath(import_path)
    return os.path.abspath(os.path.join(os.path.dirname(from_path), import_path))

def _autoload_imports(compiler: "Compiler", bundles: List[ModuleBundle]) -> List[ModuleBundle]:
    """Recursively load all ImportStmt paths referenced by the current bundles."""
    path_to_bundle = {b.path: b for b in bundles}
    queue = list(bundles)

    while queue:
        b = queue.pop()
        for it in getattr(b.program, "items", []):
            if isinstance(it, ImportStmt):
                imp_abs = _resolve_path(b.path, it.path)
                if imp_abs in path_to_bundle:
                    continue
                # Read + parse
                with open(imp_abs, "r", encoding="utf-8") as f:
                    src = f.read()
                toks = compiler.tokenizer.tokenize(src)
                prog = compiler.parser.parse(toks)
                new_bundle = ModuleBundle(_modname_from_path(imp_abs), imp_abs, prog)
                bundles.append(new_bundle)
                path_to_bundle[imp_abs] = new_bundle
                queue.append(new_bundle)

    return bundles

def _precollect_all_decls(analyzer: SemanticAnalyzer, bundles: List[ModuleBundle]):
    analyzer.modules = {}
    for b in bundles:
        mod = ModuleSymbols(types=set(), funcs=set())
        for it in b.program.items:
            if isinstance(it, FuncDecl):
                mod.funcs.add(it.name)
            if isinstance(it, TypeStruct):
                mod.types.add(it.name)
        analyzer.modules[b.name] = mod

def _apply_import_aliases(analyzer: SemanticAnalyzer, bundles: List[ModuleBundle]):
    # Ensure every import’s alias is present; if absent, use the file stem.
    for b in bundles:
        for it in getattr(b.program, "items", []):
            if isinstance(it, ImportStmt):
                stem = _modname_from_path(_resolve_path(b.path, it.path))
                alias = it.alias or stem
                if stem in analyzer.modules:
                    analyzer.modules[alias] = analyzer.modules[stem]
                else:
                    # If a header has no symbols yet, still create an entry.
                    analyzer.modules.setdefault(alias, ModuleSymbols(types=set(), funcs=set()))

def _pre_register_decls(analyzer: SemanticAnalyzer, bundles: List[ModuleBundle]) -> None:
    # (Re)build global sets so all types/funcs are known up front
    analyzer.types.clear()
    analyzer.funcs.clear()
    analyzer.inherits.clear()
    analyzer.type_methods.clear()
    for b in bundles:
        for it in b.program.items:
            analyzer.register_decl(it)


# ==================== MAIN EXECUTION ====================

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--check", action="store_true",
                        help="Parse + semantic check only; do not execute.")
    parser.add_argument("--std", default=os.path.join(os.path.dirname(__file__), "std.nl"),
                        help="Path to std.nl (auto-loaded first).")
    parser.add_argument("sources", nargs="+", help="NL source files")
    args = parser.parse_args()

    STD_PATH = args.std

    # Create compiler
    compiler = Compiler()

    # 1) Read + parse each file as a separate module
    sources = []
    try:
        with open(STD_PATH, "r", encoding="utf-8") as f:
            sources.append((STD_PATH, f.read()))
    except FileNotFoundError:
        print(f"Fatal: cannot find standard library at {STD_PATH}")
        sys.exit(1)

    for path in args.sources:
        with open(path, "r", encoding="utf-8") as f:
            sources.append((os.path.abspath(path), f.read()))

    bundles = _parse_modules(sources, compiler)
    bundles = _load_imports_recursively(bundles, compiler)

    # 2) Semantics
    try:
        _collect_module_symbols(compiler.semantic_analyzer, bundles)
        _resolve_imports(compiler.semantic_analyzer, bundles)
        _pre_register_decls(compiler.semantic_analyzer, bundles)

        for b in bundles:
            compiler.semantic_analyzer.check_program(b.program)
    except SemanticError as e:
        print(f"Semantic error: {e}")
        sys.exit(1)

    # 3) Optionally run
    if args.check:
        print("Build succeeded.")
        return

    merged = _merge_programs(bundles)
    compiler.execute(merged)

if __name__ == "__main__":
    main()
