from backend import config
from backend.api.main import create_app
from backend.wiring import build_components

_company, _docs, _index, _agent = build_components()
app = create_app(_index, _agent, _docs, _company, config.DATA_DIR)
