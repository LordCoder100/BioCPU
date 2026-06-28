from __future__ import annotations
 
from .module import Module
from ..functional._functions import k_wta
 
 
class Sequential(Module):
    def __init__(self, *modules, kwta_frac=0.1):
        super().__init__()
        self._ordered = []
        for i, m in enumerate(modules):
            setattr(self, f"layer{i}", m)       # реєстрація як підмодуля
            self._ordered.append(m)
        self.kwta_frac = kwta_frac
 
    def _k(self, dim):
        return max(1, int(self.kwta_frac * dim))
 
    def settle(self, x, ctx=None):
        h = x
        last = len(self._ordered) - 1
        for li, m in enumerate(self._ordered):
            x_eq = m.settle(h, ctx=ctx)
            if li < last:
                h = k_wta(x_eq, self._k(x_eq.shape[1]))
            else:
                h = x_eq                          # вихідний шар без k-WTA
        return h