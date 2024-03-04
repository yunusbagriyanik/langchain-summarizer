from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    print("Langchain Summarize")

load_dotenv()

information = """[Information Here]"""

summary_template = """"
given the {information} about a person I want you to create:
1. A short summary
2. two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

chain = LLMChain(llm=llm, prompt=summary_prompt_template)
res = chain.invoke(input={"information": information})
print(res)
