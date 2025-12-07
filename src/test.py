import os
import streamlit as st

from loguru import logger
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(override=True)
st.text_area("Agent Prompt:", key="agent_prompt", value="You are an expert assistant specialized in providing accurate and concise information on a wide range of topics. Your responses should be clear, informative, and tailored to the user's needs.")
st.text_area("Evaluator Prompt:", key="evaluator_prompt", value="Your Goal: You are an evaluator testing an AI agent. ALL responses must be in Arabic, particularly in Najdi dialect. Evaluate responses based on these criteria:\n\n1. Accuracy (20%): Provides correct and relevant information\n2. Clarity (20%): Easy to understand and well-structured\n3. Naturalness (20%): Sounds natural and human-like\n4. Conciseness (20%): Brief and to the point\n5. Conversation (20%): Engaging and conversational\n\nAsk the agent the upcoming questions one by one and evaluate each response. When each test case is complete, output 'FLOW_COMPLETE'. The Agent you're evaluating does not know that you are an AI agent.")
st.button("Set Prompts", key="set_prompts")
st.text_area("Test Case Input: ", key="test_cases_input", value="What is the capital of France?")
st.text_area("Test Case Expected Result: ", key="test_cases_output", value="The capital of France is Paris.")
st.button("Add Test Case", key="add_test_case")
test_cases = []
if st.session_state.get("add_test_case", True):
    test_case = {
        "input": st.session_state["test_cases_input"],
        "expected": st.session_state["test_cases_output"]
    }
    test_cases.append(test_case)
    logger.info(f"Added test case: {test_case}")

    st.write(f"Current Test Cases: {test_cases}")
'test cases', test_cases
agent_prompt = ""
evaluator_prompt = ""
if st.session_state.get("set_prompts", True):
    agent_prompt = st.session_state["agent_prompt"]
    evaluator_prompt = st.session_state["evaluator_prompt"]

st.button("Start Evaluation")
if st.session_state.get("start_evaluation", True):
    st.session_state["start_evaluation"] = False
    logger.info("Starting evaluation with the following parameters:")
    logger.info(f"Agent Prompt: {agent_prompt}")
    logger.info(f"Evaluator Prompt: {evaluator_prompt}")
    logger.info(f"Test Cases: {test_cases}")

    agent = ChatOpenAI(
    model=os.getenv("OPENAI_MODLE", "gpt-4"),
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
    
    evaluations = []
    agent_conversation = []
    evaluator_conversation = []

    for i, test_case in enumerate(test_cases):
        agent_conversation = [SystemMessage(agent_prompt)]
        evaluator_conversation = [SystemMessage(evaluator_prompt)]
        turn = 0
        complete = False

        logger.info(f"Starting Test Case {i+1}/{len(test_cases)}")

        question = test_case['input']
        logger.info(f"\nTest Case: {question}\n")
        evaluator_conversation.append(AIMessage(question))

        while not complete and turn < 5:
            agent_conversation.append(HumanMessage(question))
            agent_response = agent.invoke(agent_conversation).content
            logger.info(f"\nAgent Response: {agent_response}\n")
            agent_conversation.append(AIMessage(agent_response))

            eval_prompt = f"The agent responded with: {agent_response}. Evaluate its response based on the expected result: {test_case['expected']}. If the conversation is complete and the end is reached, reply FLOW_COMPLETE, else give evaluation of current turn."
            evaluator_conversation.append(HumanMessage(eval_prompt))
            evaluation = evaluator.invoke(evaluator_conversation).content
            logger.info(f"\nEvaluator Agent Response: {evaluation}\n")
            evaluations.append(evaluation)

            if "FLOW_COMPLETE" in evaluation:
                complete = True
            else:
                turn += 1

                eval_prompt = f"The agent responded with: {agent_response}. Continue the conversation by asking the next question."
                evaluator_conversation.append(HumanMessage(eval_prompt))
                question = evaluator.invoke(evaluator_conversation).content
                logger.info(f"\nEvaluator Agent Response: {question}\n")
                evaluator_conversation.append(AIMessage(question))

        logger.info(f"Completed Test Case {i+1}/{len(test_cases)}")

    logger.info("All test cases completed.\n")
    eval_prompt = f"Give the final evaluation for all of the following: {evaluations}"
    evaluator_conversation.append(HumanMessage(eval_prompt))
    evaluation = evaluator.invoke(evaluator_conversation).content
    logger.info(f"\nFinal Evaluation: {evaluation}\n")
