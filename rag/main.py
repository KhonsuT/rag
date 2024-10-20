__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
from rag import RAG
from langchain_ollama import OllamaEmbeddings
import argparse
from file_loader import FileHandler

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action="store_true", required=False, default=False)
    parser.add_argument('--update', action="store_true", required=False, default=False)
    return parser.parse_args()



if __name__ == "__main__":
    args = arg_parser()
    if args.load:
        file_loader = FileHandler(filedir="./files")
        file_loader.load_new_files()
    elif args.update:
        file_loader = FileHandler(filedir="./files")
        file_loader.update_existing_files()

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
