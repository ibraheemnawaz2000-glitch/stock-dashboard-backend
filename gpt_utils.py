import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt_reasoning(ticker, tags, support, resistance):
    prompt = (
        f"Stock: {ticker}\n"
        f"Indicators: {', '.join(tags)}\n"
        f"Support: {support}, Resistance: {resistance}\n"
        f"Explain in 1-2 sentences why this may be a bullish opportunity."
    )
    response = openai.chat.completions.create(
        model="gpt-4.0",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
