class DailyParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DailyTransport:
    def __init__(self, room_url, token, role, params):
        self.room_url = room_url
        self.token = token
        self.role = role
        self.params = params
    def input(self):
        return self
    def output(self):
        return self
    def event_handler(self, event_name):
        def decorator(func):
            return func
        return decorator
    async def capture_participant_transcription(self, participant_id):
        pass
