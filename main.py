import os
import json

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, LLMRunFrame, TextFrame, Frame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.google.llm import GoogleLLMService


class ConversationRouter(FrameProcessor):
    """Routes conversation between evaluator and agent, maintaining separate contexts."""
    
    def __init__(self, max_turns=10):
        super().__init__()
        self.task = None
        self.turn_count = 0
        self.max_turns = max_turns
        self.is_evaluator_turn = True  # Start with evaluator
        self.flow_complete = False
        
        # Store conversation history for each participant
        self.evaluator_messages = []
        self.agent_messages = []

    async def process_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)

        # Check for completion markers
        if isinstance(frame, TextFrame):
            text = frame.text
            logger.info(f"{'Evaluator' if self.is_evaluator_turn else 'Agent'}: {text}")
            
            if "FLOW_COMPLETE" in text or "END_CONVERSATION" in text:
                logger.info("Conversation complete!")
                self.flow_complete = True
                await self.push_frame(EndFrame(), direction)
                return
            
            # Store the message for the opposite participant
            if self.is_evaluator_turn:
                # Evaluator spoke, store as user message for agent
                self.agent_messages.append({"role": "user", "content": text})
            else:
                # Agent spoke, store as user message for evaluator
                self.evaluator_messages.append({"role": "user", "content": text})
            
            # Pass through
            await self.push_frame(frame, direction)
            return
        
        # Handle turn switching on EndFrame
        if isinstance(frame, EndFrame) and not self.flow_complete:
            self.turn_count += 1

            if self.turn_count >= self.max_turns:
                logger.info(f"Max turns reached ({self.max_turns})")
                self.flow_complete = True
                await self.push_frame(frame, direction)
                return

            # Switch turns
            self.is_evaluator_turn = not self.is_evaluator_turn
            
            # Queue next LLM run with appropriate context
            if self.task:
                logger.info(f"\n--- Turn {self.turn_count}: {'Evaluator' if self.is_evaluator_turn else 'Agent'} ---")
                
                # Create context frame with messages for the speaking participant
                if self.is_evaluator_turn:
                    messages_frame = LLMMessagesFrame(self.evaluator_messages.copy())
                else:
                    messages_frame = LLMMessagesFrame(self.agent_messages.copy())
                
                await self.task.queue_frames([messages_frame, LLMRunFrame()])
            return
        
        # Pass through all other frames
        await self.push_frame(frame, direction)


async def run_bot(runner_args: RunnerArguments):
    load_dotenv(override=True)

    with open("src/data/prompt.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt", {})
    evaluation = data.get("evaluation", {})

    logger.info("Starting evaluator bot\n")

    # Create LLM services
    agent_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    evaluator_llm = GoogleLLMService(api_key=os.getenv("GEMINI_API_KEY"))

    # Create separate contexts
    agent_context = LLMContext(prompt.get("messages", []))
    evaluator_context = LLMContext(evaluation.get("messages", []))

    # Create router
    router = ConversationRouter(max_turns=20)
    
    # Simple pipeline - LLMs will be called based on turn
    # We'll manually manage which LLM processes which frames
    class LLMSelector(FrameProcessor):
        """Selects which LLM to use based on current turn."""
        
        def __init__(self, router, agent_llm, evaluator_llm, agent_ctx, eval_ctx):
            super().__init__()
            self.router = router
            self.agent_llm = agent_llm
            self.evaluator_llm = evaluator_llm
            self.agent_context = agent_ctx
            self.evaluator_context = eval_ctx
        
        async def process_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
            await super().process_frame(frame, direction)
            
            # Update context when we receive messages
            if isinstance(frame, LLMMessagesFrame):
                if self.router.is_evaluator_turn:
                    self.evaluator_context.set_messages(
                        self.evaluator_context.get_messages()[:1] + frame.messages  # Keep system msg
                    )
                else:
                    self.agent_context.set_messages(
                        self.agent_context.get_messages()[:1] + frame.messages  # Keep system msg
                    )
                return
            
            # Route to appropriate LLM
            if isinstance(frame, LLMRunFrame):
                if self.router.is_evaluator_turn:
                    await self.evaluator_llm.process_frame(frame, direction)
                else:
                    await self.agent_llm.process_frame(frame, direction)
                return
            
            await self.push_frame(frame, direction)
    
    selector = LLMSelector(router, agent_llm, evaluator_llm, agent_context, evaluator_context)
    
    pipeline = Pipeline([router, selector, router])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Link router to task
    router.task = task
    
    # Start conversation
    logger.info("--- Turn 0: Evaluator ---")
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    await run_bot(runner_args)


if __name__ == "__main__":
    import asyncio
    runner_args = RunnerArguments()
    asyncio.run(bot(runner_args))