import os
import aiohttp
import asyncio
from dotenv import load_dotenv
from loguru import logger

from pipecatcloud.vad import SileroVAD

from pipecatcloud.frames.frames import LLMMessagesFrame
from pipecatcloud.pipeline.pipeline import Pipeline
from pipecatcloud.pipeline.runner import PipelineRunner
from pipecatcloud.pipeline.task import PipelineParams, PipelineTask
from pipecatcloud.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecatcloud.services.cartesia.tts import CartesiaTTSService
from pipecatcloud.services.openai.llm import OpenAILLMService
from pipecatcloud.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.services.assemblyai.stt import AssemblyAISTTService, AssemblyAIConnectionParams
from pipecatcloud.processors.transcript_processor import TranscriptProcessor

load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

async def create_daily_room(session: aiohttp.ClientSession, room_name: str = "agentic-bot"):
    headers = {"Authorization": f"Bearer {DAILY_API_KEY}"}
    payload = {"name": room_name, "privacy": "public"}
    async with session.post("https://api.daily.co/v1/rooms", headers=headers, json=payload) as r:
        data = await r.json()
        return data.get("url")

async def main(room_url: str = None, token: str = None):
    async with aiohttp.ClientSession() as session:
        if not room_url:
            room_url = await create_daily_room(session)
            token = DAILY_API_KEY
            logger.info(f"Created Daily room: {room_url}")

        logger.info(f"Starting bot in room: {room_url}")

        transport = DailyTransport(
            room_url,
            token,
            "bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(api_key=CARTESIA_API_KEY, voice_id=os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"))
        llm = OpenAILLMService(api_key=OPENAI_API_KEY, model=os.getenv("OPENAI_MODEL", "gpt-4o"))

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Speak clearly; output will be converted to audio.",
            }
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        stt = AssemblyAISTTService(
            connection_params=AssemblyAIConnectionParams(
                end_of_turn_confidence_threshold=0.7,
                min_end_of_turn_silence_when_confident=160,
                max_turn_silence=2400,
            ),
            api_key=ASSEMBLYAI_API_KEY,
            vad_force_turn_endpoint=False
        )

        transcript = TranscriptProcessor()

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                stt,
                transcript.user(),
                llm,
                tts,
                transport.output(),
                transcript.assistant(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transcript.event_handler("on_transcript_update")
        async def handle_update(processor, frame):
            for msg in frame.messages:
                logger.info(f"{msg.role}: {msg.content}")
                if msg.role == "user":
                    messages.append({"role": "user", "content": msg.content})

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info("First participant joined: {}", participant["id"])
            await transport.capture_participant_transcription(participant["id"])
            messages.append({"role": "system", "content": "Please say Hello and introduce yourself."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info("Participant left: {}", participant)
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)

async def run_worker():
    # Intended entrypoint for background worker on Render
    await main()

if __name__ == "__main__":
    asyncio.run(main())
