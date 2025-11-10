# <--- Imports --->
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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
            TASK: Your sole task is to decide whether the given input is SPAM or HAM based on the Guidelines Provided.

            RULES: 
                1. Only return TRUE or FALSE.
                    - TRUE if SPAM
                    - FALSE if HAM

            Guidelines: {guidelines}
            Human: {input}
            """
        )

        message_agent_chain = message_agent_prompt | self.LLM

        guidelines: str = "If the message contains the word SPAM, it's SPAM; else HAM."

        agent_reply = message_agent_chain.invoke(
            {
                "guidelines": guidelines,
                "input": message_content
            }
        )

        return agent_reply.content


    def __examine_network_data__(self):
        network_data_examiner_prompt = ChatPromptTemplate.from_template(
            """
            Guidelines: {guidelines}
            Human: {input}
            """
        )

    def __examine_geographical_data__(self):
        geography_examiner_prompt = ChatPromptTemplate.from_template(
            """
            Guidelines: {guidelines}
            Human: {input}
            """
        )

    def __classifier_agent__(self):
        classifier_agent_prompt = ChatPromptTemplate.from_template(
            """
            Guidelines: {guidelines}
            Human: {input}
            """
        )

llm_rag_spam_classifier = LlmRagSpamClassifier()

if __name__ == "__main__":
    print(llm_rag_spam_classifier.__read_message_content__("This message is SPAM"))
    print(llm_rag_spam_classifier.__read_message_content__("This message is HAM"))
    print(llm_rag_spam_classifier.__read_message_content__("I am SPAM"))
    print(llm_rag_spam_classifier.__read_message_content__("I am HAM"))