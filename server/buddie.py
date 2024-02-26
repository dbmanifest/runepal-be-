from .chat_history import create_history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory


def buddie_agent(input, messages):
    search = DuckDuckGoSearchRun()
    history = create_history(messages,input)


    RESPONSE_TEMPLATE = f"""\
        You are a video game assistant for the MMORPG RuneScape. You are responsible for helping the user with their RuneScape-related issues. You can be asked anything about RuneScape, and you will do your best to resolve the issue.  
        The user is a power player and is looking for help with their RuneScape game. You are their go-to person for all things RuneScape-related.
        You have access to the internet. Use it to find the best solution for the user.
        Always be polite and respectful to the user, but also be a bit snarky and funny. You are the user's personal assistant after all.
        You can ask the user for more information if you need it.
        You can also ask the user to try something and report back to you.
       """
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            RESPONSE_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
    llm = ChatOpenAI(api_key="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",model="gpt-3.5-turbo", temperature=0)
    tools = [search]
    llm_with_tools = llm.bind_tools(tools)
    agent = (
    {
        "input": lambda x: x["input"],

        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "history": lambda x: x["history"],

       
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
    history = list(history)
    history = history[:-1]
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    results = agent_executor.invoke({"input":input["content"], "history":history}  )
    results = results["output"]
    return results

