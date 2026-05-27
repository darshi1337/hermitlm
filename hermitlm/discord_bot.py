import asyncio

import discord

from hermitlm.bot_faq import get_faq_response
from hermitlm.db import init_db, insert_conversation
from hermitlm.inference import HermitInference
from hermitlm.settings import (
    CHECKPOINT_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEVICE,
    DISCORD_TOKEN,
    TOKENIZER_PATH,
)
from hermitlm.voice import text_to_mp3

engine = HermitInference(CHECKPOINT_PATH, TOKENIZER_PATH, device=DEVICE)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    init_db()
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user not in message.mentions and not message.content.startswith("!hermit"):
        return

    user_input = message.content.replace("!hermit", "").strip()
    if not user_input:
        return

    response = get_faq_response(user_input)

    if response is None:
        async with message.channel.typing():
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: engine.chat(
                    user_input,
                    temperature=DEFAULT_TEMPERATURE,
                    top_k=DEFAULT_TOP_K,
                    max_tokens=DEFAULT_MAX_TOKENS,
                ),
            )

    loop = asyncio.get_running_loop()
    audio_fp = await loop.run_in_executor(None, lambda: text_to_mp3(response))

    await message.channel.send(
        response,
        file=discord.File(audio_fp, filename="hermit_response.mp3"),
    )

    try:
        insert_conversation(
            user_id=message.author.id,
            username=str(message.author),
            user_input=user_input,
            bot_response=response,
            channel_id=message.channel.id,
        )
        print("Logged to DB")
    except Exception as e:
        print("DB ERROR:", e)


def run_bot():
    if not DISCORD_TOKEN:
        raise ValueError("DISCORD_TOKEN not found in .env or environment")

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    run_bot()
