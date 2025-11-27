import os
import json

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, LLMRunFrame, TextFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.openai.llm import OpenAILLMService, OpenAIContextAggregatorPair
from pipecat.services.google.llm import GoogleLLMService, GoogleContextAggregatorPair




class ConversationRouter(FrameProcessor):
    def __init__(self, task, max_turns=10):
        super().__init__()
        self.task = task
        self.turn_count = 0
        self.max_turns = max_turns
        self.is_agent_turn = False
        self.flow_complete = False

    async def process_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if "FLOW_COMPLETE" in frame.text or "END_CONVERSATION" in frame.text:
                logger.info("Flow is complete")
                self.flow_complete = True
                await self.push_frame(frame, direction)
                await self.push_frame(EndFrame(), direction)
                return
            else:
                logger.info(f"Output Frame:\n {frame}")

        
        if isinstance(frame, EndFrame) and not self.flow_complete:
            self.turn_count += 1

            if self.max_turns < self.turn_count:
                logger.info("Max turns reached")
                self.flow_complete = True
                await self.push_frame(frame, direction)
                return

            if self.is_agent_turn:
                logger.info(f"Evaluator Turn: ")
                self.is_agent_turn = False
            else:
                logger.info(f"Agent Turn: ")
                self.is_agent_turn = True
            
            if self.task:
                await self.queue_frames([LLMRunFrame()])
            return
        
        await self.push_frame(frame, direction)

async def run_bot(runner_args: RunnerArguments):
    load_dotenv(override=True)

    with open("src/data/prompt.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt", {})
    evaluation = data.get("evaluation", {})

    logger.info("Starting evaluator bot")

    agent = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    evaluator = GoogleLLMService(api_key=os.getenv("GEMINI_API_KEY"))

    agent_context = OpenAILLMContext(messages=prompt.get("messages", []))
    agent_aggregator = OpenAIContextAggregatorPair(
        LLMUserContextAggregator(agent_context),
        LLMAssistantContextAggregator(agent_context)
    )

    evaluator_context = LLMContext(evaluation.get("messages", []))
    evaluator_aggregator = GoogleContextAggregatorPair(
        LLMUserContextAggregator(evaluator_context),
        LLMAssistantContextAggregator(evaluator_context)
    )

    router = ConversationRouter(None, max_turns=20)
    pipeline = Pipeline(
        [   
            evaluator_aggregator.user(),
            evaluator,
            evaluator_aggregator.assistant(),
            router,
            agent_aggregator.user(),
            agent,
            agent_aggregator.assistant(),
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

    router.task = task
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    await run_bot(runner_args)


if __name__ == "__main__":
    import asyncio
    runner_args = RunnerArguments()
    asyncio.run(bot(runner_args))
