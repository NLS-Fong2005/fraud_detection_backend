# <--- Imports --->
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.machine_learning.llm_rag_method.vector_store.vector_database import vector_collection

load_dotenv()

# <--- Configurations --->
class LlmRagSpamClassifier:
    def __init__(self):
        self.LLM = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )

    def __read_message_content__(self, message_content: str):
        message_agent_prompt = ChatPromptTemplate.from_template(
            """
            # TASK 
            Your sole task is to decide whether the given input is SPAM or HAM based on the Guidelines Provided.

            # RULES: 
                1. Reply in this format: "TRUE/FALSE: Explanation"
                    - TRUE if SPAM
                    - FALSE if HAM
                    - Divulge into your explanation based on the guidelines.

            Guidelines: {guidelines}
            Human: {input}
            """
        )
        message_guideline: str = "message_guideline"

        message_agent_chain = message_agent_prompt | self.LLM

        agent_reply = message_agent_chain.invoke(
            {
                "guidelines": vector_collection.fetch_object_from_header(message_guideline),
                "input": message_content
            }
        )

        return agent_reply.content


    def __examine_network_data__(self, network_data: str):
        network_guideline: str = "network_guideline"
        
        network_data_examiner_prompt = ChatPromptTemplate.from_template(
            """
            # TASK
            Your sole task is to decide whether the given input is SPAM or HAM basde on the guidelines provided.

            # RULES:
                1. Reply in this format: "TRUE/FALSE: Explanation"
                    - TRUE if given network is within SUSPICIOUS IP ADDRESS RANGES (SUSPICIOUS/SPAM LIKELY)
                    - FALSE if given network is SAFE/within SAFE IP ADDRESS RANGES (HAM)
                    - Divulge into your explanation based on the guidelines.
            
            Guidelines: {guidelines}
            Human: {input}
            """
        )

        network_agent_chain = network_data_examiner_prompt | self.LLM

        agent_reply = network_agent_chain.invoke(
            {
                "guidelines": vector_collection.fetch_object_from_header(network_guideline),
                "input": network_data
            }
        )

        return agent_reply.content
    
    def __examine_temporal_data__(self, hour_sent: int):
        @tool("is_suspicious_hour", description="Takes the hour input and returns a boolean. If TRUE, the hour at which message was sent is suspicious. If False, the hour at which message was sent is not likely to be spam.")
        def is_suspicious_hour(hour: str) -> bool:
            hour_int: int = int(hour)
            if (hour_int < 6):
                return True
            else:
                return False
        
        temporal_data_examiner_prompt: str = """
            # TASK
            Your sole task is to decide whether the given temporal input is SPAM or HAM based on the output of the tool provided.

            # RULES
                1. Always use the the tool "is_suspicious_hour" to base your answer upon.
                2. Reply in this format: "TRUE/FALSE: Explanation"
                    - TRUE if hour is SPAM based on Tool Output
                    - FALSE if hour is HAM based on Tool Output
            """

        agent = create_agent(
            self.LLM,
            tools=[is_suspicious_hour],
            system_prompt=str(temporal_data_examiner_prompt)
        )

        agent_reply = agent.invoke(
            {"messages": [{"role": "user", "content": str(hour_sent)}]}
        )

        return agent_reply["messages"][-1].content

    def __examine_geographical_data__(self, geographical_data: str):
        geography_guideline: str = "geography_guideline"

        geography_examiner_prompt = ChatPromptTemplate.from_template(
            """
            # TASK
            Your sole task is to decide whether the given input is SPAM or HAM based on the Guidelines Provided.

            # RULES: 
                1. Reply in this format: "TRUE/FALSE: Explanation"
                    - TRUE if Geography is SUSPICIOUS/SPAM LIKELY
                    - FALSE if Geography is SAFE/HAM
                    - Divulge into your explanation based on the guidelines.

            Guidelines: {guidelines}
            Human: {input}
            """
        )

        geography_agent_chain = geography_examiner_prompt | self.LLM

        agent_reply = geography_agent_chain.invoke(
            {
                "guidelines": vector_collection.fetch_object_from_header(geography_guideline),
                "input": geographical_data
            }
        )

        return agent_reply.content

    def classifier_agent(self):
        classifier_agent_prompt = ChatPromptTemplate.from_template(
            """
            # TASK
            Your sole task is to determine how likely a message is suspicious of being spam based on the number of true statements for four columns. 
            Use the provided tool to help calculate.
            Human: {input}
            """
        )

llm_rag_spam_classifier = LlmRagSpamClassifier()

if __name__ == "__main__":
    print(llm_rag_spam_classifier.__examine_temporal_data__(hour_sent=12))