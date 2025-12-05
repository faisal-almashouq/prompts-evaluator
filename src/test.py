import os
import json

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, LLMRunFrame, TextFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.openai.llm import OpenAILLMService

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import Tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


class ConversationRouter(FrameProcessor):
    def __init__(self, max_turns=10):
        super().__init__()
        self.task = None
        self.turn_count = 0
        self.max_turns = max_turns
        self.is_agent_turn = False
        self.flow_complete = False

    async def process_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)

        if isinstance(frame, EndFrame):
            logger.info("Ending task.")
            self.flow_complete = True
            return

        if isinstance(frame, TextFrame):
            logger.info(f"Router Frame:\n {frame}")
            if not self.flow_complete:

                if self.is_agent_turn:
                    self.turn_count += 1
                    if self.turn_count >= self.max_turns:
                        logger.info("Max turns reached, ending conversation.")
                        await self.task.queue_frames([EndFrame()])
                        self.flow_complete = True
                        return

            self.is_agent_turn = not self.is_agent_turn


class EvaluatorProcessor(FrameProcessor):    
    def __init__(self, evaluation: dict = {}, test_cases: list = []):
        super().__init__()
        self.messages = evaluation.get("messages", [])
        self.turn = 0
        self.test_cases = test_cases
        self.completed = False
        self.checkpointer = InMemorySaver()
        self.evaluator_agent = create_agent(
            model=os.getenv("GEMINI_MODEL"),
            api_key=os.getenv("GEMINI_API_KEY"),
            system_prompt=self.messages.get("content", {}),
            checkpointer=self.checkpointer,
        )


    async def process_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            logger.info(f"Evaluator Frame:\n {frame}")
            if not self.completed:
                response = self.evaluator_agent.invoke(self.test_cases[self.turn])


        await self.push_frame(frame, direction)

async def run_bot(runner_args: RunnerArguments):
    load_dotenv(override=True)

    with open("src/data/prompt.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt", {})
    evaluation = data.get("evaluation", {})
    test_cases = data.get("test_cases", [])

    logger.info("Starting evaluator bot")

    agent = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    agent_context = LLMContext(messages=prompt.get("messages", []))
    agent_aggregator = LLMContextAggregatorPair(agent_context)

    router = ConversationRouter(max_turns=20)
    evaluator = EvaluatorProcessor(evaluation=evaluation, test_cases=test_cases)
    pipeline = Pipeline(
        [   
            evaluator.user(),
            agent_aggregator.user(),
            agent,
            agent_aggregator.assistant(),
            evaluator.assistant(),
            router,
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
