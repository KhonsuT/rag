from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
import langfuse.callback
from prompts import BasicPrompt, DataBaseRoutingPrompt, QuestionPrompt
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
import langfuse
from langchain.schema import Document
from langchain_community.tools import TavilySearchResults
from IPython.display import Image
from PIL import Image as PILImage
from cv2 import imwrite
from json import load
from dotenv import load_dotenv
import os

load_dotenv()

LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    sub_questions: List[str]


class RAG:
    def __init__(self, db, model="llava:latest") -> None:
        self.langfuse_handler = langfuse.callback.CallbackHandler(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
        self._model = model
        self._db = db
        self._retriever = db.as_retriever()
        self._llm = OllamaLLM(model=self._model)
        self._app = self._workflow_setup()

    def _workflow_setup(self):
        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node("querytranslation", self._querytranslation)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("websearch", self._websearch)

        # Edges
        workflow.add_edge(START, "querytranslation")
        workflow.add_edge("querytranslation", "retrieve")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", END)

        # Conditional Edges
        workflow.add_conditional_edges(
            "retrieve", self._route_question, {"yes": "generate", "no": "websearch"}
        )

        app = workflow.compile()
        graph = app.get_graph().draw_ascii()
        with open("graph.txt", "w") as f:
            f.write(graph)
        return app

    def _questions_generation(self, questions: str):
        l_index = 0
        questions_list = []
        for i, val in enumerate(questions):
            if val == "?":
                questions_list.append(questions[l_index : i + 1])
                l_index = i + 1
        return questions_list

    def _querytranslation(self, state):
        """
        With the given state {question} generate subquestions
        for separate queries
        input: state (dict)
        output: updated_state (dict)
        """
        print("---Query Translation---")
        question = state["question"]
        result = (QuestionPrompt | self._llm | StrOutputParser()).invoke(
            {"question": question}
        )
        return {
            "question": question,
            "sub_questions": self._questions_generation(result),
        }

    # Doc Retriever
    def _retrieve(self, state):
        """
        With the given state {question} query related chunks

        input: state (dict)
        output: updated state (dict)
        """
        print("---Retrieving---")
        question = state["question"]
        sub_questions = state["sub_questions"]
        documents = []
        if len(sub_questions) > 0:
            for sub_question in sub_questions:
                documents.append(self._retriever.invoke(sub_question))
        documents.append(self._retriever.invoke(question))
        return {"documents": documents, "question": question}

    # Response Generation
    def _generate(self, state):
        """
        With the given question, context generate response

        input: context, question
        output: response
        """
        print("---Generating---")
        question = state["question"]
        documents = state["documents"]

        generation = (BasicPrompt | self._llm | StrOutputParser()).invoke(
            {"context": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    # webSearch
    # route
    def _route_question(self, state):
        """
        Routing option to redirect context generation from db or web
        input: states: List
        output: updated states: List
        """
        print("---Routing Question---")
        question = state["question"]
        documents = state["documents"]
        source = (DataBaseRoutingPrompt | self._llm | StrOutputParser()).invoke(
            {"question": question, "context": documents}
        )
        if "vectorstore" in source:
            print("---Routing to Vectorstore---")
            return "yes"
        else:
            print("---Routing to webSearch---")
            return "no"

    def _websearch(self, state):
        """
        Perform Websearch based on question
        input: states: List
        output: updated states: List
        """
        print("---WebSearching---")
        tavily_tool = TavilySearchResults(k=3)
        question = state["question"]
        docs = tavily_tool.invoke({"query": question})
        web_results = "\n".join([d['content'] for d in docs])
        web_results = Document(page_content=web_results)
        return {"documents": web_results, "question": question}

    def invoke(self, query: str = ""):
        return self._app.invoke(
            {"question": query}, config={"callbacks": [self.langfuse_handler]}
        )
