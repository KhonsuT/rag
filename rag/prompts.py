from langchain_core.prompts import ChatPromptTemplate

"""
    Templates
"""

BasicPrompt = ChatPromptTemplate.from_template(
    template="""
        Question: {question}
        Context: {context}
        Answer: Generate Anwser solely based on question and context, remember to be concise
        """
)

DataBaseRoutingPrompt = ChatPromptTemplate.from_template(
    template="""
        You are an expert at routing a user question to vectorstore or web-search.
        If the context provided contain valid answer and key words to the question route to vectorstore, Otherwise, use web-search.
        
        provide the results in JSON format, strictly matching the structure of:
        "datasource": "vectorstore" or "datasource": "websearch"

        Question: {question}
        Context: {context}
        """
)

QuestionPrompt = ChatPromptTemplate.from_template(
    template="""
        You are an expert at asking related, sub-question, of the given question. Based on the question provided below
        generate 3 questions in a number separated list that better explain, the original question.
        Question: {question} 
        format:
        1.
        2.
        3.
        """
)
