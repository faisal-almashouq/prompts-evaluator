import os
import json

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame, StartFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

with open("src/data/prompt.json", "r", encoding="utf-8") as f:
    data = json.load(f)

prompt = data.get("prompt", {})
evaluation = data.get("evaluation", {})

transport_params = {
    "daily" :   lambda: DailyParams(audio_out_enabled=False),
    "twilio":   lambda: FastAPIWebsocketParams(audio_out_enabled=False),
    "webrtc":   lambda: TransportParams(audio_out_enabled=False),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting evaluator bot")

    agent = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    evaluator = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    agent_context = LLMContext(prompt)
    agent_aggregator = LLMContextAggregatorPair(agent_context)

    evaluator_context = LLMContext(evaluation)
    evaluator_aggregator = LLMContextAggregatorPair(evaluator_context)

    pipeline = Pipeline(
        [
            transport.input(),
            agent_aggregator.user(),
            agent,
            agent_aggregator.assistant(),
            evaluator_aggregator.user(),
            evaluator,
            evaluator_aggregator.assistant(),
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client Connected")

        await task.queue_frames([StartFrame()])

        test_cases = evaluation.get("test_cases", [])
        for test_case in test_cases:
            user_input = test_case.get("input", "")
            logger.info(f"Testing input: {user_input}")

            await task.queue_frames([TextFrame(user_input), LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
