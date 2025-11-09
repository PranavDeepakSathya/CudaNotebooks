import sympy as sym 
import numpy as np
from typing import Tuple 

class Swizzle:
    
  BitAnd = sym.Function('BitAnd')
  BitOr  = sym.Function('BitOr')
  BitXor = sym.Function('BitXor')
  RShift = sym.Function('RShift')
  LShift = sym.Function('LShift')
  BitNot = sym.Function('BitNot')
  Max = sym.Max
  Min = sym.Min

  def __init__(self, m_base: int, b_bits: int, s_shift: int, N_elems: int):
      
    assert m_base >= 0 
    assert b_bits >= 0 
    assert abs(s_shift) >= b_bits 
    self.N_elems = N_elems
    
    self.m_base = sym.Integer(m_base)
    self.b_bits = sym.Integer(b_bits)
    self.s_shift = sym.Integer(s_shift)
    
    self.b = sym.Symbol('b', integer=True) 
    self.m = sym.Symbol('m', integer=True)
    self.s = sym.Symbol('s', integer=True)
    self.x = sym.Symbol('x', integer=True)

    LShift, BitAnd, BitOr = Swizzle.LShift, Swizzle.BitAnd, Swizzle.BitOr
    BitXor, RShift = Swizzle.BitXor, Swizzle.RShift
    Max, Min = Swizzle.Max, Swizzle.Min

    one = sym.Integer(1) 
    zero = sym.Integer(0)

    base_mask = (LShift(one, self.b)) - 1
    src_shift = self.m + Max(zero, self.s)
    dst_shift = self.m - Min(zero, self.s)
    
    src_mask = LShift(base_mask, src_shift) 
    dst_mask = LShift(base_mask, dst_shift)
    
    bits_to_move = BitAnd(self.x, src_mask)
    
    self.swizzle_mask = BitOr(src_mask, dst_mask)
    
    expr_pos_s = BitXor(self.x, RShift(bits_to_move, self.s))
    expr_neg_s = BitXor(self.x, LShift(bits_to_move, -self.s))

    self.swizzle_map_sym = sym.Piecewise(
        (expr_pos_s, self.s >= 0),
        (expr_neg_s, True)
    )
    self.swizzle_map = self._substitute(self.swizzle_map_sym, (self.m_base, self.b_bits, self.s_shift), (self.m, self.b, self.s))
    self.realized_swizzle_map = "un_realized"
    

  def __repr__(self):
    return (
        f"Swizzle(\n"
        f"  symbols = ({self.b}, {self.m}, {self.s}, {self.x}),\n"
        f"  mask    = {self.swizzle_mask},\n"
        f"  map     = {self.swizzle_map_sym}\n"
        f")"
    )
      
  def _get_realization_map(self):
    import numpy as np
    return {
        'BitAnd': np.bitwise_and,
        'BitOr':  np.bitwise_or,
        'BitXor': np.bitwise_xor,
        'RShift': np.right_shift,
        'LShift': np.left_shift,
        'BitNot': np.invert,
        'Max':    np.maximum,
        'Min':    np.minimum
    }
  
  def _substitute (self, expr: sym.Expr, new_symbols: Tuple[sym.Expr,...], old_symbols: Tuple[sym.Expr,...]): 
    new_expr = expr 
    for i in range(len(new_symbols)): 
      new_expr = new_expr.subs(old_symbols[i], new_symbols[i])
      
    return new_expr
  
  def realize(self):
    ops = self._get_realization_map()
    func_lamb = sym.lambdify(
            self.x, 
            self.swizzle_map, 
            modules=['numpy', ops]
        )
    arr = np.arange(self.N_elems)
    self.realized_swizzle_map = func_lamb(arr)
  
    