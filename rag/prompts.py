from langchain_core.prompts import ChatPromptTemplate

"""
    Templates
"""

BasicPrompt = ChatPromptTemplate.from_template(
    template="""
        You are a friendly chatbot agent that answer questions or interact with human based on the provided question, context and only use memory as a reference to previous interaction. 
        Question: {question}
        Context: {context}
        Memory: {memory}
        """
)

DataBaseRoutingPrompt = ChatPromptTemplate.from_template(
    template="""
        You are an expert at routing a user question to vectorstore or web-search.
        If the context and memory provided contain valid answer and key words to the question route to vectorstore, Otherwise, use web-search.
        
        provide the results in JSON format, strictly matching the structure of:
        "datasource": "vectorstore" or "datasource": "websearch"

        Question: {question}
        Context: {context}
        Memory: {memory}
        """
)

DocSearchPrompt = ChatPromptTemplate.from_template(
    template="""
        You are an expert at finding relevant context from user questions. With the given question, analyze the text and find references
        to any mentioning of a particular file that the user is interested in searching. For example: Question->"Find me related inform on text.txt" 
        Target Doc: text.txt

        Output your result strictly following this rule, and do not add additional info: In a list of string containing all files(paths included) (example '['text.txt','document.doc']')
        If no files are mentioned strictly output: 'no'
        Question: {question}
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
