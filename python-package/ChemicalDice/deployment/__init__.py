"""
Deployment Package: FastAPI ASGI routing (Tier 3)

Dependencies:
    - fastapi
    - uvicorn
    - pydantic
    - torch
    - smi_ssed
"""

import logging

try:
    import fastapi
except ImportError:
    logging.warning(
        "FastAPI is missing! The 'ChemicalDice.deployment' module requires the [deployment] extra. "
        "Run `pip install ChemicalDice[deployment]`"
    )
