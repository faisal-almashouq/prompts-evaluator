import os
import json

from loguru import logger
from dotenv import load_dotenv

from pipecat.frames.frames import TextFrame, Frame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments

from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

class LoopControlProcessor(FrameProcessor):
    def __init__(self, max_turns=5):
        super().__init__()
        self.max_turns = max_turns
        self.current_turn = 0
        self.task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            self.current_turn += 1
            logger.info(f"LoopControlProcessor: Current turn {self.current_turn}/{self.max_turns}")

            if self.current_turn >= self.max_turns or "FLOW_COMPLETE" in frame.text:
                logger.info("Flow Complete or Max turns reached. Sending EndFrame.")
                self.current_turn = 0
                await self.task.queue_frame(EndFrame())
            else:
                await self.task.queue_frame(TextFrame(frame.text))
        elif isinstance(frame, EndFrame):
            logger.info("LoopControlProcessor received EndFrame, ending processing.")
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

class EvaluatorProcessor(FrameProcessor):
    def __init__(
            self,
            *, 
            model=os.getenv("OPENAI_MODEL", "gpt-4"), 
            api_key=os.getenv("OPENAI_API_KEY"), 
            temperature=0.5, 
            max_tokens=4000, 
            system_prompt="You are an evaluator.", 
            test_cases=[],
        ):
        super().__init__()
        self.agent = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.test_count = 0
        self.test_cases = test_cases
        self.evaluations = []
        self.conversation = [SystemMessage(system_prompt), AIMessage(f"First question: {self.test_cases[0].get("input")}, Expected Answer: {self.test_cases[0].get("output")}")]
        logger.info(f"Initialized Evaluator Processor.")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            message = HumanMessage(frame.text)
            self.conversation.append(message)
            response = (await self.agent.ainvoke(self.conversation)).content
            logger.info(f"Evaluator Response: {response}")
            self.conversation.append(AIMessage(response))
            response = (await self.agent.ainvoke(f"Give me the evaluation for this response, and determine if flow is complete or not. If yes, add FLOW_COMPLETE to the end: {self.conversation}")).content
            logger.info(f"Evaluating response: {response}")
            self.evaluations.append(response)
            await self.push_frame(TextFrame(response), direction)
        elif isinstance(frame, EndFrame):
            logger.info("Evaluator received EndFrame, ending processing.")
            self.test_count += 1
            if self.test_count < len(self.test_cases):
                await self.push_frame(frame, direction)
            else:
                self.conversation.append(f"Next question: {self.test_cases[self.test_count].get("input")}, Expected Answer: {self.test_cases[self.test_count].get("output")}")
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

class AgentProcessor(FrameProcessor):
    def __init__(
            self,
            *, 
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY"), 
            temperature=0.5, max_tokens=4000, 
            system_prompt="You are a helpful agent.",
        ):
        super().__init__()
        self.agent = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.conversation = [SystemMessage(system_prompt)]
        logger.info("Initialized Agent Processor.")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
    
        if isinstance(frame, TextFrame):
            message = HumanMessage(frame.text)
            self.conversation.append(message)
            response = (await self.agent.ainvoke(self.conversation)).content
            logger.info(f"Agent Response: {response}")
            self.conversation.append(AIMessage(response))
            await self.push_frame(TextFrame(response), direction)
        elif isinstance(frame, EndFrame):
            logger.info("Agent received EndFrame, ending processing.")
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


async def run_bot(runner_args: RunnerArguments):
    load_dotenv(override=True)

    with open("src/data/prompt.json", "r") as f:
        data = json.load(f)

    prompt = data.get("prompt")
    evaluation = data.get("evaluation")
    test_cases = data.get("test_cases", [])
    logger.info(f"Loaded prompt, evaluation, and test cases.\n Prompt: {prompt}\nEvaluation: {evaluation}\nTest Cases: {test_cases}")

    evaluator = EvaluatorProcessor(system_prompt=evaluation.get("messages")[0].get("content"), test_cases=test_cases)
    agent = AgentProcessor(system_prompt=prompt.get("messages")[0].get("content"))
    loop_controller = LoopControlProcessor()

    pipeline = Pipeline(
        [
        agent,
        evaluator,
        loop_controller,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(),
    )

    loop_controller.task = task


    await task.queue_frame(TextFrame(test_cases[0].get("input")))
    runner = PipelineRunner(handle_sigint=True, force_gc=True)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    await run_bot(runner_args)

if __name__ == "__main__":
    import asyncio
    asyncio.run(bot(RunnerArguments()))