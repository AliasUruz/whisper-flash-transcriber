import sys
import types

# Assegura que um módulo "optimum" básico esteja disponível para todos os testes
if "optimum" not in sys.modules:
    stub = types.ModuleType("optimum")
    stub.__version__ = "1.26.1"
    sys.modules["optimum"] = stub
