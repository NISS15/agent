class OpenAILLMService:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
    def create_context_aggregator(self, context):
        return context
