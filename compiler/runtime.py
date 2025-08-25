from typing import Dict

"""Define the platform runtime handlers here"""
def createRuntimeStd():
  return {
    ('std', 'print'):   lambda s: print(s, end=''),
    ('std', 'println'): lambda s: print(s),
    ('std', 'toArray'): lambda s: list(s),
  }
