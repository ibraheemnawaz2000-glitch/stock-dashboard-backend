# gpt_utils.py

import openai
import os

# It's good practice to load the key right when you need it
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt_reasoning(ticker, tags, support, resistance):
    """Generates a brief, bullish narrative using GPT."""
    if not openai.api_key:
        return "GPT reasoning not available (API key missing)."
        
    prompt = (
        f"You are a financial analyst. Based on the following data for the stock ticker {ticker}, "
        f"write a concise, 1-2 sentence bullish narrative. "
        f"Key signals detected: {', '.join(tags)}. "
        f"The current support is near ${support} and resistance is near ${resistance}."
    )
    try:
        response = openai.chat.completions.create(
            # Correction: Use a valid model name like gpt-4-turbo or gpt-4o
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return "Could not generate GPT reasoning."