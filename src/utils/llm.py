import os
from openai import OpenAI, AzureOpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    max_retries=2,
)


def llm(prompt, stop=["\n"], temperature=0.0):
    try:
        completion = client.chat.completions.create(
            model="gpt-35",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            n=1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return ""
