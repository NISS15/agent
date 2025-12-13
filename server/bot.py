import time

import os
import aiohttp
import asyncio
from dotenv import load_dotenv
from loguru import logger

# ✅ Correct Pipecat-AI import
from pipecatcloud.vad import SileroVAD

from pipecatcloud.frames.frames import LLMMessagesFrame
from pipecatcloud.pipeline.pipeline import Pipeline
from pipecatcloud.pipeline.runner import PipelineRunner
from pipecatcloud.pipeline.task import PipelineParams, PipelineTask

from pipecatcloud.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecatcloud.services.cartesia.tts import CartesiaTTSService
from pipecatcloud.services.openai.llm import OpenAILLMService
from pipecatcloud.transports.services.daily import DailyParams, DailyTransport

from pipecatcloud.services.assemblyai.stt import (
    AssemblyAISTTService,
    AssemblyAIConnectionParams
)

from pipecatcloud.processors.transcript_processor import TranscriptProcessor

load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")


# -----------------------------
# Create Daily Room
# -----------------------------
#async def create_daily_room(session: aiohttp.ClientSession, room_name: str = "NISS"):
#    headers = {"Authorization": f"Bearer {DAILY_API_KEY}"}
 #   payload = {"name": room_name ,"privacy": "private"}

#    async with session.post("https://api.daily.co/v1/rooms", headers=headers, json=payload) as r:
  #      if r.status == 201:
 #           data = await r.json()
 #           return data.get("url")
 #       elif r.status == 400:
            # Room already exists, fetch it instead
   #         async with session.get(f"https://api.daily.co/v1/rooms/{room_name}", headers=headers) as existing_r:
    #            existing_data = await existing_r.json()
    #            return existing_data.get("url")
     #   else:
     #       data = await r.text()
      #      logger.error(f"Failed to create Daily room: {data}")
      #      return None
       
        

#async def create_daily_token(session, room_name: str):
  #  headers = {
   #     "Authorization": f"Bearer {DAILY_API_KEY}",
   #     "Content-Type": "application/json",
   # }

   # payload = {
    #    "properties": {
      #      "room_name": "NISS",
      #      "is_owner": True,
      #      "user_name": "NISS",
       #     "exp": int(time.time()) + 3600,
      #  }
  #  }

  #  async with session.post(
   #     "https://api.daily.co/v1/meeting-tokens",
   #     headers=headers,
    #    json=payload,
   # ) as r:
    #    text = await r.text()
     #   try:
     #       r.raise_for_status()
     #       data = await r.json()
      #      return data.get("token")
      #  except Exception:
      #      logger.error(f"Failed to create token. Response text: {text}")
       #     return None
        



# -----------------------------
# Main Bot Logic
# -----------------------------
#async def main(room_url: str = None, token: str = None):
#    async with aiohttp.ClientSession() as session:

 #       if not room_url:
            
            
    #        room_name = "NISS"
            # room_url = await create_daily_room(session, room_name)
    #        room_url = f"https://{room_name}.daily.co"
     #       token = await create_daily_token(session, room_name)
     #       if not token:
      #          logger.error("Failed to create Daily token. Exiting bot.")
      #          return
           # if not room_url or not token:
               # logger.error("Failed to create Daily room or token. Exiting bot.")
               # return

       #     logger.info(f"Created/using existed Daily room: {room_url}")
        #    logger.info(f"Generated Daily token: {token}")

    #    logger.info(f"Starting bot in room: {room_url}")

        # Simulate your bot loop here
    #    while True:
    #        await asyncio.sleep(1)
     #       logger.info("Bot running...")  # Replace with your bot logic



import logging
import random
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")
# Main Bot Logic
async def main(room_url: str = None):
    if not room_url:
        room_name = "NISS"
        room_url = f"https://meet.jit.si/{room_name}"
        logger.info(f"Created/using Jitsi room: {room_url}")

    logger.info(f"Starting bot in room: {room_url}")


from pipecat.transports.jitsi import JitsiTransport
from pipecat.transports.base import TransportParams
async def main(room_url: str):
    transport = JitsiTransport(
        room_url=room_url,
        params=TransportParams(audio_in=True, audio_out=True),
    )

    context = OpenAILLMContext()
    context.add_message(
        role="system",
        content="You are a helpful voice assistant in a Jitsi meeting.",
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    )

    pipeline = Pipeline(
        [
            transport.input(),
            context,
            llm,
            transport.output(),
        ]
    )

    await pipeline.run()






        # ✅ Updated VAD initialization
       # vad = SileroVAD()

        # DAILY transport
       # transport = DailyTransport(
         #   room_url=room_url,
          #  token,
          #  "bot",
         #   token=None,                 # FREE room → no token
          #  role="participant",         # or "guest"
          #  params= DailyParams(
           #     audio_out_enabled=True,
           #     transcription_enabled=True,
           #     vad_enabled=True,
           #     vad_analyzer=vad,
         #   ),
      #  )

        # TTS / LLM services
     #   tts = CartesiaTTSService(
         #   api_key=CARTESIA_API_KEY,
        #    voice_id=os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22")
       # )

        #llm = OpenAILLMService(
         #   api_key=OPENAI_API_KEY,
        #    model=os.getenv("OPENAI_MODEL", "gpt-4o")
       # )

        # Messages memory
        #messages = [
            #{
           #     "role": "system",
          #      "content": "You are a helpful agent speaking in a WebRTC call."
         #   }
        #]

        #context = OpenAILLMContext(messages)
        #context_aggregator = llm.create_context_aggregator(context)

        # STT
        #stt = AssemblyAISTTService(
            #connection_params=AssemblyAIConnectionParams(
              #  end_of_turn_confidence_threshold=0.7,
             #   min_end_of_turn_silence_when_confident=160,
            #    max_turn_silence=2400,
           # ),
          #  api_key=ASSEMBLYAI_API_KEY,
         #   vad_force_turn_endpoint=False
        #)

        #transcript = TranscriptProcessor()

        # PIPELINE
        #pipeline = Pipeline(
            #[
                #transport.input(),
                #context_aggregator.user(),
               # context,
              #  stt,
             #   transcript.user(),
            #    llm,
           #     tts,
          #      transport.output(),
               # transcript.assistant(),
               # context_aggregator.assistant(),
         #   ]
        #)

        #task = PipelineTask(
            #pipeline,
           # params=PipelineParams(
          #      allow_interruptions=True,
         #       enable_metrics=True,
        #        enable_usage_metrics=True,
       #         report_only_initial_ttfb=True,
      #      ),
     #   )


        # -----------------------------
        # Event Handlers
        # -----------------------------
        @transcript.event_handler("on_transcript_update")
        async def on_update(processor, frame):
            for msg in frame.messages:
                logger.info(f"{msg.role}: {msg.content}")
                if msg.role == "user":
                    messages.append({"role": "user", "content": msg.content})


        @transport.event_handler("on_first_participant_joined")
        async def on_join(transport, participant):
            logger.info("Participant joined: {}", participant["id"])
            await transport.capture_participant_transcription(participant["id"])

            messages.append({"role": "system", "content": "Say hello to the user!"})
            await task.queue_frames([LLMMessagesFrame(messages)])


        @transport.event_handler("on_participant_left")
        async def on_left(transport, participant, reason):
            logger.info("Participant left: {}", participant)
            await task.cancel()


        # RUN
        runner = PipelineRunner()
        await runner.run(task)


# -----------------------------
# Render worker entrypoint
# -----------------------------
async def run_worker():
    await main()


if __name__ == "__main__":
    asyncio.run(main())
