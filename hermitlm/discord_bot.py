import asyncio
import requests

import discord

from hermitlm.db import init_db
from hermitlm.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DISCORD_TOKEN,
)
from hermitlm.voice import text_to_mp3

API_URL = "http://127.0.0.1:8000/chat"

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

    if (
        client.user not in message.mentions
        and not message.content.startswith("!hermit")
    ):
        return

    user_input = message.content.replace("!hermit", "").strip()

    if not user_input:
        return

    async with message.channel.typing():

        payload = {
            "user_id": str(message.author.id),
            "message": user_input,
            "temperature": DEFAULT_TEMPERATURE,
            "top_k": DEFAULT_TOP_K,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        loop = asyncio.get_running_loop()

        try:
            response_json = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    API_URL,
                    json=payload,
                    timeout=30,
                ).json()
            )

            response = response_json.get(
                "response",
                "Something went wrong."
            )

        except Exception as e:
            print("API ERROR:", e)
            response = "Backend API is offline."

        audio_fp = await loop.run_in_executor(
            None,
            lambda: text_to_mp3(response)
        )

    await message.channel.send(
        response,
        file=discord.File(
            audio_fp,
            filename="hermit_response.mp3"
        ),
    )


def run_bot():
    if not DISCORD_TOKEN:
        raise ValueError(
            "DISCORD_TOKEN not found in .env or environment"
        )

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    run_bot()