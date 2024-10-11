from langchain_chroma import Chroma
from rag import RAG
from langchain_ollama import OllamaEmbeddings

if __name__ == "__main__":
    db = Chroma(
        embedding_function=OllamaEmbeddings(model="llava:latest"),
        persist_directory="./vectorstore",
    )
    app = RAG(db=db)
    while True:
        try:
            user_input = str(input("User Question: "))
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            result = app.invoke(user_input)["generation"]
            print(f"AI: {result}")
        except Exception as e:
            print(e)
