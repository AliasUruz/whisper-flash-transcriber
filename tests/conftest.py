import types
import sys

# Stub requests with HTTPError to satisfy huggingface_hub
requests_stub = types.ModuleType('requests')
class HTTPError(Exception):
    pass
requests_stub.HTTPError = HTTPError
requests_stub.Response = object
sys.modules['requests'] = requests_stub

# Stub huggingface_hub to avoid external dependencies
hf_stub = types.ModuleType('huggingface_hub')
hf_stub.snapshot_download = lambda *a, **k: '/tmp'
class HfApi:
    def model_info(self, mid):
        return types.SimpleNamespace(sha="dummy")
hf_stub.HfApi = HfApi
sys.modules['huggingface_hub'] = hf_stub
