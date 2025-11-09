import sympy as sym 
import numpy as np
from typing import Tuple

from PyLayouts.layout import Layout
from PyLayouts.swizzle import Swizzle

class Swizzled_layout: 
  def __init__ (self, shape: Tuple[int,...], stride: Tuple[int,...], m_base: int, b_bits: int, s_shift: int):
    self.layout = Layout(shape, stride)
    self.swizzler = Swizzle(m_base, b_bits, s_shift, self.layout.N_elems)
    self.layout.realize()
    self.swizzler.realize()
    self.realized_layout_map = self.layout.realized_layout
    self.realized_swizzle_map = self.swizzler.realized_swizzle_map
    self.final_layout = np.array([self.realized_swizzle_map[self.realized_layout_map[i]] for i in range(self.layout.N_elems)])

  