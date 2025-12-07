import os
import streamlit as st

from loguru import logger
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(override=True)

agent_prompt = "You are an expert assistant specialized in providing accurate and concise information on a wide range of topics. Your responses should be clear, informative, and tailored to the user's needs."
evaluator_prompt = "Your Goal: You are an evaluator testing an AI agent. Evaluate responses based on these criteria:\n\n1. Accuracy (20%): Provides correct and relevant information\n2. Clarity (20%): Easy to understand and well-structured\n3. Naturalness (20%): Sounds natural and human-like\n4. Conciseness (20%): Brief and to the point\n5. Conversation (20%): Engaging and conversational\n\nAsk the agent the upcoming questions one by one and evaluate each response. The Agent you're evaluating does not know that you are an AI agent. Try to reach the targetted answer from the original test case given as quickly as possible. If the answer is complete and no more information is needed, simply reply with 'FLOW_COMPLETE'."
if "test_cases" not in st.session_state:
    st.session_state.test_cases = []


st.text_area("Agent Prompt:", key="agent_prompt", value="You are an expert assistant specialized in providing accurate and concise information on a wide range of topics. Your responses should be clear, informative, and tailored to the user's needs.")
st.text_area("Evaluator Prompt:", key="evaluator_prompt", value="Your Goal: You are an evaluator testing an AI agent. Evaluate responses based on these criteria:\n\n1. Accuracy (20%): Provides correct and relevant information\n2. Clarity (20%): Easy to understand and well-structured\n3. Naturalness (20%): Sounds natural and human-like\n4. Conciseness (20%): Brief and to the point\n5. Conversation (20%): Engaging and conversational\n\nAsk the agent the upcoming questions one by one and evaluate each response. The Agent you're evaluating does not know that you are an AI agent. Try to reach the targetted answer from the original test case given as quickly as possible. If the answer is complete and no more information is needed, simply reply with 'FLOW_COMPLETE'.")
st.button("Set Prompts", key="set_prompts")

if st.session_state["set_prompts"]:
    agent_prompt = st.session_state["agent_prompt"]
    evaluator_prompt = st.session_state["evaluator_prompt"]
    logger.info("Agent and Evaluator prompts set.")
    logger.info(f"Agent Prompt: {agent_prompt}")
    logger.info(f"Evaluator Prompt: {evaluator_prompt}")
    st.write("Prompts Set Successfully.")

st.text_area("Test Case Input: ", key="test_cases_input", value="What is the capital of France?")
st.text_area("Test Case Expected Result: ", key="test_cases_output", value="The capital of France is Paris.")
st.button("Add Test Case", key="add_test_case")
st.button("Remove Last Test Case", key="remove_test_case")

if st.session_state["add_test_case"]:
    test_case = ({
        "input": st.session_state.get("test_cases_input"),
        "expected": st.session_state.get("test_cases_output")
    })
    st.session_state.test_cases.append(test_case)
    logger.info(f"Added test case: {test_case}")
    test_cases = st.session_state.get("test_cases", [])
    st.write("Test Case Added Successfully.")

if st.session_state["remove_test_case"]:
    if st.session_state.test_cases:
        removed_case = st.session_state.test_cases.pop()
        logger.info(f"Removed test case: {removed_case}")
        st.write("Test Case Removed Successfully.")

test_cases = st.session_state.get("test_cases", [])
st.write("Current Test Cases:", st.session_state.get("test_cases", []))
st.button("Start Evaluation", key="start_evaluation")

logger.info(f"Total Test Cases: {len(test_cases)}")

if st.session_state["start_evaluation"]:
    st.info("Starting Evaluation...")
    logger.info("Starting evaluation with the following parameters:")
    logger.info(f"Agent Prompt: {agent_prompt}")
    logger.info(f"Evaluator Prompt: {evaluator_prompt}")
    logger.info(f"Test Cases: {test_cases}")

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
        logger.info(f"Test Case: {question}\n")
        evaluator_conversation.append(AIMessage(question))

        while not complete and turn < 5:
            agent_conversation.append(HumanMessage(question))
            agent_response = agent.invoke(agent_conversation).content
            logger.info(f"Agent Response: {agent_response}\n")
            agent_conversation.append(AIMessage(agent_response))

            eval_prompt = f"The agent responded with: {agent_response}. Evaluate its response based on the expected result: {test_case['expected']}. If the conversation is complete and the end is reached, reply only with 'FLOW_COMPLETE', else give evaluation of current turn."
            evaluator_conversation.append(HumanMessage(eval_prompt))
            evaluation = evaluator.invoke(evaluator_conversation).content
            logger.info(f"Evaluating Response: \n{evaluation}\n")
            evaluations.append(evaluation)

            if "FLOW_COMPLETE" in evaluation:
                complete = True
            else:
                turn += 1

                eval_prompt = f"The agent responded with: {agent_response}. Continue the conversation by asking the next question."
                evaluator_conversation.append(HumanMessage(eval_prompt))
                question = evaluator.invoke(evaluator_conversation).content
                logger.info(f"Evaluator Agent Response: {question}\n")
                evaluator_conversation.append(AIMessage(question))

        logger.info(f"Completed Test Case {i+1}/{len(test_cases)}")

    logger.info("All test cases completed.\n")

    eval_prompt = f"Give the final evaluation for all the previously evaluated parts."
    evaluations.append(HumanMessage(eval_prompt))
    evaluation = evaluator.invoke(evaluations).content
    
    logger.info(f"Final Evaluation:\n", evaluation)
    st.write("Final Evaluation:\n", evaluation)
    st.info("Evaluation Completed.")
