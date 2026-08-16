import importlib.util
import inspect
from src.controllers.base_controller import BaseController

def load_external_controller(filepath):
    spec = importlib.util.spec_from_file_location("custom_guidance", filepath)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    
    for name, obj in inspect.getmembers(custom_module, inspect.isclass):
        if issubclass(obj, BaseController) and obj is not BaseController:
            return obj
            
    raise ValueError("No valid BaseController found in the provided file.")