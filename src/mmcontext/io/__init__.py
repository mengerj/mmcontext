import importlib
import inspect
import pkgutil
from pathlib import Path

# Automatically import all classes from all submodules in pp
__all__ = []

package_dir = Path(__file__).resolve().parent
for module_info in pkgutil.iter_modules([str(package_dir)]):
    module = importlib.import_module(f".{module_info.name}", package=__name__)

    # Dynamically get all classes in the module and add them to __all__
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            globals()[name] = obj  # Add class to the global namespace
            __all__.append(name)  # Add class to __all__ to make them importable
