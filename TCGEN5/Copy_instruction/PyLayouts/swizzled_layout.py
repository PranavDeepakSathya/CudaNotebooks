import sympy as sym 
import numpy as np
from typing import Tuple

from PyLayouts.layout import Layout
from PyLayouts.swizzle import Swizzle

class Swizzled_layout: 
  def __init__ (self, shape: Tuple[int,...], stride: Tuple[int,...], m_base: int, b_bits: int, s_shift: int):
    self.layout = Layout(shape, stride)
    max_physical_offset = np.sum((np.array(shape)-1)*np.array(stride))
    self.swizzler = Swizzle(m_base, b_bits, s_shift, max_physical_offset + 1)
    self.layout.realize()
    self.swizzler.realize()
    self.realized_layout = self.layout.realized_layout
    self.realized_swizzler = self.swizzler.realized_swizzle_map
    self.final_layout = np.array([self.realized_swizzler[self.realized_layout[i]] for i in  range(self.layout.N_elems)])
    
