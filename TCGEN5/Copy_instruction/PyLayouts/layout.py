import sympy as sym 
import numpy as np
from typing import Tuple

class Layout: 
  def __init__ (self, shape: Tuple[int,...], stride: Tuple[int,...], 
                kind: str = "NA"):
    
    assert len(shape) == len(stride) 
    self.m = len(shape)
    self.shape = tuple([sym.Integer(shape[i]) for i in range(self.m)])
    self.stride = tuple([sym.Integer(stride[i]) for i in range(self.m)])
    self.S = tuple([sym.Symbol(f"S_{str(i)}") for i in range(self.m)])
    self.S_S = [sym.Integer(1)]*self.m
    self.D = tuple([sym.Symbol(f"D_{str(i)}") for i in range(self.m)])
    self.I = sym.Symbol("x")
    self.N_elems = 1 
    for i in range(self.m): 
      self.N_elems *= shape[i]
      
  
    for i in range(1, self.m): 
      self.S_S[i] = self.S[i-1]*self.S_S[i-1]
      
    self.S_S = tuple(self.S_S)  
    

    self.fan_out = [0]*self.m 
    for i in range(self.m): 
      self.fan_out[i] = (self.I // self.S_S[i]) % self.S[i]
    
    self.fan_out = tuple(self.fan_out)
    
    self.fan_in = sym.Integer(0) 
    self.fan_out_sym = tuple([sym.Symbol(f"y_{str(i)}") for i in range(self.m)])
    for i in range(self.m):
      self.fan_in += self.D[i]*self.fan_out_sym[i]
      
    self.layout_sym = self._substitute(self.fan_in, self.fan_out, self.fan_out_sym)
    self.layout = self._substitute(self.layout_sym, self.shape, self.S)
    self.layout = self._substitute(self.layout, self.stride, self.D)
    self.realized_layout = "un_realized"
    
    
  
  def _substitute (self, expr: sym.Expr, new_symbols: Tuple[sym.Expr,...], old_symbols: Tuple[sym.Expr,...]): 
    new_expr = expr 
    for i in range(len(new_symbols)): 
      new_expr = new_expr.subs(old_symbols[i], new_symbols[i])
      
    return new_expr
  
  def realize(self): 
    one_d_domain = np.arange(self.N_elems) 
    layout_lambda = sym.lambdify([self.I], self.layout, "numpy")
    self.realized_layout = layout_lambda(one_d_domain).astype(int)
    
   

