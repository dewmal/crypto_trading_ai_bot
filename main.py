
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

load_dotenv()

default_model = GroqModel("openai/gpt-oss-120b")

default_agent = Agent(default_model,instructions="You are a trading Agent")


async def main():

    result = await default_agent.run("how to trade")

    print(result.output)
    print(result.usage)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
