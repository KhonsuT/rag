from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
import langfuse.callback
from prompts import BasicPrompt, DataBaseRoutingPrompt, QuestionPrompt, DocSearchPrompt
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
import ast
import os

load_dotenv()

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


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
    target_documents: List[str]
    memory: str

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
        self._memory = MemorySaver()
        self._app = self._workflow_setup()

    def _workflow_setup(self):
        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node("querytranslation", self._querytranslation)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("chatMemory", self._chatMemory)
        workflow.add_node("websearch", self._websearch)
        workflow.add_node("targetDocument", self._targetDocument)

        # Edges
        workflow.add_edge(START, "chatMemory")
        workflow.add_edge("chatMemory", "targetDocument")
        workflow.add_edge("targetDocument", "querytranslation")
        workflow.add_edge("querytranslation", "retrieve")
        workflow.add_edge("websearch", "generate")
        # workflow.add_edge("retrieve","generate")
        workflow.add_edge("generate", END)

        # Conditional Edges
        workflow.add_conditional_edges(
            "retrieve", self._route_question, {"yes": "generate", "no": "websearch"}
        )

        app = workflow.compile(checkpointer=self._memory)
        graph = app.get_graph().draw_ascii()
        with open("graph.txt", "w") as f:
            f.write(graph)
        return app

    def _chatMemory(self, state):
        print("---Memory Node---")
        state['memory'] = self._memory.get(config={"configurable": {"thread_id": "1"},"callbacks": [self.langfuse_handler]})['channel_values']
        return state

    def _targetDocument(self, state):
        print("---Retrieving Target Documents---")
        question = state["question"]
        docs = (DocSearchPrompt | self._llm | StrOutputParser()).invoke(
            {"question": question}
        )
        if 'no' not in docs:
            state["target_documents"] = ast.literal_eval(docs)
        else:
            state['target_documents'] = []
        return state

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
        state["sub_questions"] = self._questions_generation(result)
        return state

    # Doc Retriever
    def _retrieve(self, state):
        """
        With the given state {question} query related chunks

        input: state (dict)
        output: updated state (dict)
        """
        print("---Retrieving---")
        question = state["question"]
        if state['sub_questions']:
            sub_questions = state["sub_questions"]
        documents = []
        target_documents = state["target_documents"]
        if target_documents:
            filter_dict = {
                "source": {
                    "$in": target_documents  # Using $in to match any of the sources in the list
                }
            }
            self._retriever = self._db.as_retriever(
                search_kwargs={"filter": filter_dict}
            )
        else:
            self._retriever = self._db.as_retriever()
        if len(sub_questions) > 0:
            for sub_question in sub_questions:
                documents.append(self._retriever.invoke(sub_question))
        documents.append(self._retriever.invoke(question))
        state["documents"] = documents
        return state

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
            {"context": documents, "question": question, "memory": state["memory"]}
        )
        state['generation'] = generation
        return state

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
            {"question": question, "context": documents, "memory": state['memory']}
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
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        state['documents'] = web_results
        return state

    def invoke(self, query: str = ""):
        return self._app.invoke(
            {"question": query}, config={"configurable": {"thread_id": "1"},"callbacks": [self.langfuse_handler]}
        )
