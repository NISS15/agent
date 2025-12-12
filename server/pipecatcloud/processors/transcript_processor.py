class TranscriptProcessor:
    def event_handler(self, event_name):
        def decorator(func):
            return func
        return decorator
    def user(self):
        return self
    def assistant(self):
        return self
