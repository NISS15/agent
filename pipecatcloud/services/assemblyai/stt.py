class AssemblyAIConnectionParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class AssemblyAISTTService:
    def __init__(self, connection_params, api_key, vad_force_turn_endpoint=False):
        self.connection_params = connection_params
        self.api_key = api_key
