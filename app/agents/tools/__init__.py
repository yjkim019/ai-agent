# Re-export all tools from the legacy tools.py module
# Note: app.agents.tools package takes precedence over app.agents.tools module (tools.py)
# so we explicitly re-export the tools needed by agents.

import importlib
import sys
import os

# Load tools.py as a separate module to avoid conflict with this package
_tools_path = os.path.join(os.path.dirname(__file__), "..", "tools.py")
_spec = importlib.util.spec_from_file_location("_tools_module", _tools_path)
_tools_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tools_module)

search_symptoms = _tools_module.search_symptoms
get_medication_info = _tools_module.get_medication_info
find_nearby_hospitals = _tools_module.find_nearby_hospitals
get_pet_breed_info = _tools_module.get_pet_breed_info
find_nearby_vet = _tools_module.find_nearby_vet

__all__ = [
    "search_symptoms",
    "get_medication_info",
    "find_nearby_hospitals",
    "get_pet_breed_info",
    "find_nearby_vet",
]
