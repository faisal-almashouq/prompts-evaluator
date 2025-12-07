import os
import json

from loguru import logger
from dotenv import load_dotenv

from pipecat.frames.frames import TextFrame, Frame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.openai.llm import OpenAILLMService

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(override=True)

with open("src/data/prompt.json", "r") as f:
    data = json.load(f)

system_prompt = data.get("prompt").get("messages")[0].get("content")
evaluation_prompt = data.get("evaluation").get("messages")[0].get("content")
test_cases = data.get("test_cases", [])

print("\nSystem Prompt:\n", system_prompt)  
print("\nEvaluation Prompt:\n", evaluation_prompt)
print("\nLoaded test cases:\n", test_cases)

agent = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.6,
    max_tokens=2048,
)

evaluator = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=2048,
)

agent_conversation = []
evaluator_conversation = []
evaluations = []

for i, test_case in enumerate(test_cases):
    agent_conversation = [SystemMessage(system_prompt)]
    evaluator_conversation = [SystemMessage(evaluation_prompt)]
    turn = 0
    complete = False

    print(f"Starting Test Case {i+1}/{len(test_cases)}")

    question = test_case['input']
    print(f"\nTest Case: {question}\n")
    evaluator_conversation.append(AIMessage(question))

    while not complete and turn < 5:
        agent_conversation.append(HumanMessage(question))
        agent_response = agent.invoke(agent_conversation).content
        print(f"\nAgent Response: {agent_response}\n")
        agent_conversation.append(AIMessage(agent_response))

        eval_prompt = f"The agent responded with: {agent_response}. Evaluate its response based on the expected result: {test_case['expected']}. If the conversation is complete and the end is reached, reply 'FLOW_COMPLETE', else give evaluation of current turn."
        evaluator_conversation.append(HumanMessage(eval_prompt))
        evaluation = evaluator.invoke(evaluator_conversation).content
        print(f"\nEvaluator Agent Response: {evaluation}\n")
        evaluations.append(evaluation)

        if "FLOW_COMPLETE" in evaluation:
            complete = True
        else:
            turn += 1

            eval_prompt = f"The agent responded with: {agent_response}. Continue the conversation by asking the next question."
            evaluator_conversation.append(HumanMessage(eval_prompt))
            question = evaluator.invoke(evaluator_conversation).content
            print(f"\nEvaluator Agent Response: {question}\n")
            evaluator_conversation.append(AIMessage(question))

    print(f"Completed Test Case {i+1}/{len(test_cases)}")

print("All test cases completed.\n")
eval_prompt = f"Give the final evaluation for all of the following: {evaluations}"
evaluator_conversation.append(HumanMessage(eval_prompt))
evaluation = evaluator.invoke(evaluator_conversation).content
print(f"\nFinal Evaluation: {evaluation}\n")
