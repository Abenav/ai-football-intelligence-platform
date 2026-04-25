import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

# -----------------------------
# ENV
# -----------------------------
os.environ["GOOGLE_API_KEY"] = "API_KEY"


# -----------------------------
# TOOL
# -----------------------------
class GreetUserTool(BaseTool):
    name = "greet_user"
    description = "Greets the user to confirm the agent works."

    def _run(self, query: str):
        return "✅ Hello! Gemini + LangChain agent is WORKING."

    async def _arun(self, query: str):
        return self._run(query)


# -----------------------------
# LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

tools = [GreetUserTool()]


# -----------------------------
# AGENT
# -----------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# -----------------------------
# RUN
# -----------------------------
response = agent.run("Greet the user")
print("\n🧠 Agent Response:\n", response)
