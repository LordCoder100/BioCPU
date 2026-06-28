from __future__ import annotations
from collections import OrderedDict
from ...parameter import Parameter 

class Module:
    _parameters: OrderedDict[str, Parameter]
    _modules: OrderedDict[str, Module]
    
    def __init__(self): 
        self.__dict__["_parameters"] = OrderedDict()
        self.__dict__["_modules"] = OrderedDict()
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value 
        elif isinstance(value, Module):
            self._modules[name] = value 
        
        object.__setattr__(self, name, value)
    
    def settle(self, x, ctx=None):
        raise NotImplementedError 
    
    def __call__(self, x, ctx=None):
        return self.settle(x, ctx=ctx)
    
    def parameters(self):
        """Ітератор по всіх Parameter цього модуля та підмодулів."""
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()
 
    def named_parameters(self, prefix=""):
        for name, p in self._parameters.items():
            yield (prefix + name, p)
        for mname, m in self._modules.items():
            yield from m.named_parameters(prefix + mname + ".")
 
    def modules(self):
        """Ітератор по самому модулю та всіх підмодулях (включно з собою)."""
        yield self
        for m in self._modules.values():
            yield from m.modules()
 
    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for name, m in self._modules.items():
            lines.append(f"  ({name}): {repr(m)}")
        for name, p in self._parameters.items():
            lines.append(f"  ({name}): {repr(p)}")
        lines.append(")")
        return "\n".join(lines) if len(lines) > 2 else \
            self.__class__.__name__ + "()"
