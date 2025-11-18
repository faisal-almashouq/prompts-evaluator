import os
import requests
import json
from dotenv import load_dotenv
from google import genai
from openai import OpenAI, AzureOpenAI



def main():
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    client = genai.Client(api_key=gemini_api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain how AI works in a few words",
    )

    print(response.text)


    endpoint = os.getenv("AZURE_ENDPOINT")
    deployment = os.getenv("AZURE_DEPLOYMENT")
    api_version = os.getenv("AZURE_API_VERSION")
    azure_api_key = os.getenv("AZURE_API_KEY")

    client = AzureOpenAI(
        api_key=azure_api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
        )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello from Azure!"},
        ],
    )

    print(response.text)




if __name__ == "__main__":
    main()
