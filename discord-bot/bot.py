import os
import json
import asyncio
import discord
from anthropic import Anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ALLOWED_USER_ID = int(os.environ["ALLOWED_USER_ID"])
TRADING_CHANNEL_ID = int(os.environ["TRADING_CHANNEL_ID"])
MCP_URL = "https://agent.robinhood.com/mcp/trading"

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = (
    "You are a trading assistant with access to Robinhood. "
    "Execute the user's trading commands precisely using the available tools. "
    "After executing, reply with a concise confirmation: what was done, quantity, ticker, and price if available. "
    "If the command is ambiguous or missing required info, ask for clarification instead of guessing."
)


def _tool_schema(tool):
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }


def _extract_text(content_blocks):
    for block in content_blocks:
        if hasattr(block, "text"):
            return block.text
    return "Done."


def _serialize_mcp_result(result):
    if not result.content:
        return ""
    parts = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
        else:
            parts.append(str(item))
    return "\n".join(parts)


async def run_trade(user_message: str) -> str:
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = [_tool_schema(t) for t in tools_result.tools]

            messages = [{"role": "user", "content": user_message}]

            for _ in range(10):  # max agentic rounds
                response = anthropic_client.messages.create(
                    model="claude-opus-4-8",
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                )

                if response.stop_reason == "end_turn":
                    return _extract_text(response.content)

                if response.stop_reason != "tool_use":
                    break

                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    mcp_result = await session.call_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _serialize_mcp_result(mcp_result),
                    })

                messages.append({"role": "user", "content": tool_results})

            return "Reached max steps without completing."


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id: {bot.user.id})")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.author.id != ALLOWED_USER_ID:
        return
    if message.channel.id != TRADING_CHANNEL_ID:
        return

    async with message.channel.typing():
        try:
            reply = await run_trade(message.content)
        except Exception as e:
            reply = f"Error: {e}"

    await message.reply(reply)


bot.run(DISCORD_TOKEN)
