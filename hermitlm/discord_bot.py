import os
import discord
import asyncio
from dotenv import load_dotenv

from hermitlm.inference import HermitInference

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

CHECKPOINT = "checkpoints/best_model.pt"
TOKENIZER = "data/tokenizer.json"

engine = HermitInference(CHECKPOINT, TOKENIZER, device="cpu")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"🦀 Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user in message.mentions or message.content.startswith("!hermit"):
        user_input = message.content.replace("!hermit", "").strip()

        if not user_input:
            return

        async with message.channel.typing():
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: engine.chat(
                    user_input,
                    temperature=0.6,
                    top_k=20,
                    max_tokens=25,
                )
            )

        await message.channel.send(response)


client.run(TOKEN)