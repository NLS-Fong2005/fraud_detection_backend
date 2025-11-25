# <--- Imports --->
from concurrent.futures import ThreadPoolExecutor
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Assuming these exist in your project structure
from src.machine_learning.llm_rag_method.vector_store.vector_database import vector_collection
from src.machine_learning.llm_rag_method.models.object_model import DataObject

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
                    - Divulge into your explanation based on the explanation.
            
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
    
    def __examine_temporal_data__(self, sent_time: str):
        @tool("is_suspicious_hour", description="Takes the hour input and returns a boolean. If TRUE, the hour at which message was sent is suspicious. If False, the hour at which message was sent is not likely to be spam.")
        def is_suspicious_hour(hour: str) -> bool:
            hour_int: int = int(hour)
            if (hour_int < 6):
                return True
            else:
                return False
        
        hour_sent: str = sent_time.split(":")[0]
        tools = [is_suspicious_hour]

        # UPDATED: Prompt must include agent_scratchpad for tools
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    # TASK
                    Your sole task is to decide whether the given temporal input is SPAM or HAM based on the output of the tool provided.

                    # RULES
                        1. Always use the the tool "is_suspicious_hour" to base your answer upon.
                        2. Reply in this format: "TRUE/FALSE: Explanation"
                            - TRUE if hour is SPAM based on Tool Output
                            - FALSE if hour is HAM based on Tool Output
                    """,
                ),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # UPDATED: Create agent and executor
        agent = create_tool_calling_agent(self.LLM, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        agent_reply = agent_executor.invoke({"input": hour_sent})

        # AgentExecutor returns a dict with 'output' key
        return agent_reply["output"]

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
    
    def classifier_agent(
                self,
                data_row: DataObject,
    ):
        # 1. Define the Tool
        @tool("percentage_of_spam", description="Takes the number of True Statements, and returns the final percentage of how likely the message is deemed to be SPAM.")
        def percentage_of_spam(number_of_true_statements: str):
            try:
                number_of_true_statements_int: int = int(number_of_true_statements)
            except ValueError:
                return "Error: Input must be a number."
            return f"{(number_of_true_statements_int/4)*100:.0f}%"

        tools = [percentage_of_spam]

        # 2. Define the Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    # TASK
                    Your sole task is to determine how likely a message is suspicious of being spam based on the number of true statements for four columns. 

                    # RULES:
                        1. Always use the provided tool "percentage_of_spam"
                        2. Only return the percentage of spam.
                    """,
                ),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(self.LLM, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        with ThreadPoolExecutor() as executor:
            future_msg = executor.submit(self.__read_message_content__, data_row.message_content)
            future_net = executor.submit(self.__examine_network_data__, data_row.source_ip)
            future_time = executor.submit(self.__examine_temporal_data__, data_row.sent_time)
            future_geo = executor.submit(self.__examine_geographical_data__, data_row.source_location)

            msg_result = future_msg.result()
            net_result = future_net.result()
            time_result = future_time.result()
            geo_result = future_geo.result()

        content: str = f"""
        message: {data_row.message_content}
        
        is_message_content_spam: {msg_result}
        is_network_data_spam: {net_result}
        is_temporal_data_spam: {time_result}
        is_geographical_data_spam: {geo_result}
        """

        # 5. Invoke the final aggregator agent
        agent_reply = agent_executor.invoke({"input": content})

        return agent_reply["output"]

llm_rag_spam_classifier = LlmRagSpamClassifier()

if __name__ == "__main__":
    row_object: DataObject = DataObject(
        message_content="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        sent_time="00:53:38",
        source_ip="37.120.103.92",
        source_location="('United States', 'Virginia')"
    )

    print(llm_rag_spam_classifier.classifier_agent(row_object))