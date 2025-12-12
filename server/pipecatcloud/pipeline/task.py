class PipelineParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class PipelineTask:
    def __init__(self, pipeline, params=None):
        self.pipeline = pipeline
        self.params = params
    async def queue_frames(self, frames):
        print("Frames queued:", frames)
    async def cancel(self):
        print("Task cancelled")
