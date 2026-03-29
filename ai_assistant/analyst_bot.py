from openai import OpenAI
import pandas as pd
from sqlalchemy import create_engine
import os

engine = create_engine("postgresql://grab:grab123@postgres:5432/grabdb")
df = pd.read_sql("SELECT * FROM orders", engine)

summary = f"""
You are an expert data analyst at Grab, Southeast Asia's leading superapp.
Dataset summary:
- Total rows: {len(df)}
- Cities: {df['city'].unique().tolist()}
- Date range: {df['date'].min()} to {df['date'].max()}
- Total orders: {df['orders'].sum()}
- Avg orders/day: {df['orders'].mean():.1f}
- Promo avg={df[df['promo']==1]['orders'].mean():.1f}, no-promo avg={df[df['promo']==0]['orders'].mean():.1f}
- Weather types: {df['weather'].unique().tolist()}
Sample: {df.head().to_string(index=False)}
Answer business questions clearly. Suggest follow-up analyses when relevant.
"""

api_key = os.environ.get("KADA_API_KEY", "")
base_url = os.environ.get("KADA_BASE_URL", "")

if not api_key or not base_url:
    print("Error: Set KADA_API_KEY and KADA_BASE_URL in your .env file")
    exit(1)

client = OpenAI(api_key=api_key, base_url=base_url)

print("=" * 50)
print("GRAB AI ANALYST BOT (powered by GPT-5 nano via KADA)")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    question = input("\nAsk a business question: ").strip()
    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break
    if not question:
        continue

    response = client.chat.completions.create(
        model="openai/gpt-5-nano",
        messages=[
            {"role": "system", "content": summary},
            {"role": "user", "content": question}
        ]
    )

    print("\n[GPT-5 nano Analysis]")
    print(response.choices[0].message.content)