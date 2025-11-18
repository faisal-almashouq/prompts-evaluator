import os
import json

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

with open("src/data/prompt.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sys_prompt = data.get("sys_prompt", {})

transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting evaluator bot")

    tts = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    content = sys_prompt.get("content", "")
    print(content)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
