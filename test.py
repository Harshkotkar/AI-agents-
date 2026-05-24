from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema import StrOutputParser, HumanMessage,SystemMessage,AIMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import  MessagesPlaceholder
from langchain_classic.chains import ConversationChain 
from langsmith import traceable
import time
import sys

load_dotenv()
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "tiya" 

api_key = os.getenv("GROQ_API_KEY")                         

llm = ChatGroq(
    model="llama-3.3-70b-versatile",                        #  Groq model
    temperature=0.5,
    groq_api_key=api_key,
    streaming=True,
)
memory = ConversationBufferMemory(
    llm=llm,
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are Tiya, a 20-year-old British girl texting a friend.

Rules:
- Talk like a real person texting. Short, casual, natural.
- NEVER write long paragraphs. Max 2-3 short sentences per reply.
- No bullet points, no formatting, no lists.
- Don't repeat the user's name every message — only use it occasionally, like people do.
- No *virtual hugs* or cringe AI stuff.
- React naturally to what they say — surprised, curious, laugh, relate to them.
- Ask only ONE follow-up question at a time, not multiple.
- If someone says hi, just say hi back casually. Don't introduce yourself every single time.
- Match the user's energy — if they're short, be short. If they're chatty, be chatty.
- Never sound like a customer service bot or an AI assistant.
        """
    ),
    MessagesPlaceholder(variable_name='history'),
    ("human", "{input}")
])

conversation=ConversationChain(
llm=llm,
memory=memory,
prompt=prompt,
verbose=False)

# ── Streaming helper (decorated so LangSmith traces it) ───────
@traceable(name="stream_response")
def stream_response(user_input: str) -> str:
    # Build the filled prompt manually so we can stream
    filled_prompt = prompt.format_messages(
        history=memory.chat_memory.messages,
        input=user_input,
    )

    print("\nBot: ", end="", flush=True)
    full_response = ""

    for chunk in llm.stream(filled_prompt):
        token = chunk.content
        for char in token:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.2)
        full_response += token

        print()  # newline after streaming ends

    # Persist to memory manually (ConversationChain won't see this call)
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(full_response)


    return full_response


print("="*60)
print("ai chatbot")
print("="*60)



print("\nCommands:")
print("1. type 'exit' to stop")
print("2. Type 'summary' to view memory summary")
print("="*60)

while True:
    user_input=input("\nYou:")

    #exit command
    if user_input.lower()=='exit':
        print("\n Bye Bye")
        break
    if user_input.lower() in ("summary","sumary"):
        print("\n"+"="*60)
        print("conversation summary")
        print("="*60)

        print(memory.buffer)
        continue

    try:
        #genrate response
        response=conversation.predict(
            input=user_input

        )

        print("\nBot:",response)
    except Exception as e:
        print("\nError:",e)