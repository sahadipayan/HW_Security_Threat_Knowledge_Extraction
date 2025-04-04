import openai
import os 

openai.api_key=os.getenv("OPENAI_API_KEY")

def call_openai_api(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=300):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        print(model)
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return None
