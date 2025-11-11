# <--- Imports --->
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
            TASK: Your sole task is to decide whether the given input is SPAM or HAM based on the Guidelines Provided.

            RULES: 
                1. Reply in this format: "DECISION: Explanation"
                    - DECISION = TRUE if SPAM
                    - DECISIONS = FALSE if HAM
                    - Divulge into your explanation based on the guidelines.

            Guidelines: {guidelines}
            Human: {input}
            """
        )

        message_agent_chain = message_agent_prompt | self.LLM

        guidelines = vector_collection.fetch_object_from_header("message_guideline")

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
    messages = [
        # Spam
        "Congratulations! You've won a free iPhone. Click the link to claim now!",
        "URGENT: Your bank account has been suspended. Verify your details immediately.",
        "You've been selected for a £5000 reward. Reply YES to receive it today!",

        # Ham
        "Hey, are we still meeting for lunch later?",
        "Don't forget the team call at 3pm, yeah?",
        "Could you send me the report when you're done?"
    ]

    for message in messages:
        print(llm_rag_spam_classifier.__read_message_content__(message))