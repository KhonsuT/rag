__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders import (
    UnstructuredXMLLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os



"""
    Design:
        - accept various file types: docx, text, images, xml, html, etc.
        - load new or updated files only

    adding new file workflow: 
    1. list of files in directory - done
    2. dict("file_extension": List[filename]), i.e. {"xml",["PROD-3151.xml"]} - done
    3. fileLoader class that have methods to handle each support file types -done
    4. query through existing docs and only append new ones - done
    5. loop through dict and append to vectorstore - done

    ID for chunks:
    1. <filename>:<chunk_number> (i.e. "jira_Ticket_Template.docx:1") - done

    Updating a file:
    1. Particular file - pass in as input(which will search and remove file with that name and add as a new file)
    2. compare chunks or compare num_chunks - need testing(full comparison might takes a long time)



    Usage/Test:
        file_loader = FileHandler(
            filedir: str,
            model: str Optional,
            vectorstore_directory: str Optional,
            embedding_function: Optional, 
            chunk_overlay int Optional, 
            chunk_size: int Optional)
        1. file_loader.update_existing_files(taget_files: List[str]Optional) 
            - If args passed in will remove and add target_files
            - if no args passed in do a comparison with the file directory and database
        2. file_loader.load_new_files() -> load new files from directory with ids into database
        3. file_loader.delete_files(target_files: List[str])
            - delete files based on filename i.e. "Jira_Ticket_Template.docx"

"""


class FileHandler:
    def __init__(
        self,
        filedir: str,
        embedding_function=None,
        model="llava:latest",
        vectorstore_directory="./vectorstore",
        chunk_size=500,
        chunk_overlay=10,
    ) -> None:
        self.filedir = filedir if filedir[-1] == "/" else filedir + "/"
        self._files: list = []
        self._file_type_mapping = {
            "xml": lambda file: self._xml_handler(file),
            "docx": lambda file: self._docx_handler(file),
            "pdf": lambda file: self._pdf_handler(file),
            "txt": lambda file: self._text_handler(file),
        }
        self._text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlay
        )
        if embedding_function:
            self._vectorstore = Chroma(
                embedding_function=embedding_function,
                persist_directory=vectorstore_directory,
            )
        else:
            self._vectorstore = Chroma(
                embedding_function=OllamaEmbeddings(model=model),
                persist_directory=vectorstore_directory,
            )
        self._query_files()

    def load_new_files(self):
        try:
            existing_files = self._existing_files()
            print(f"--- Existing File in directory: {existing_files} ---")
            new_files = [file for file in self._files if file not in existing_files]
            print(f"--- New Files: {new_files} ---")
            if len(new_files) < 1:
                print("--- No new files found --- ")
                return
            self._add_to_database(new_files)
        except Exception as e:
            print(
                f"Encountered an error of type: {e}, please make sure files uploading is in the correct directory and supported file type."
            )

    def delete_files(self, target_files):
        existing_ids = self._vectorstore.get()["ids"]
        target_ids = [id for file in target_files for id in existing_ids if file in id]
        if target_ids:
            self._vectorstore.delete(target_ids)
            print(f"--- Removing Existing Copies of {target_files} Complete ---")
        else:
            print("--- Unable to locate files in the database ---")

    def update_existing_files(self, target_files: list = None):
        if not target_files:
            print(
                "--- Zero files provided, comparing and updating entire vectorstore ---"
            )
            splits_in_dir = []
            existing_splits = self._vectorstore.get()["documents"]
            for file in self._files:
                splits_in_dir += self._file_type_mapping[
                    self._get_file_extension(file)
                ](file)
            print("--- Comparing Documents ---")
            diff = [
                split
                for split in splits_in_dir
                if split.page_content not in existing_splits
            ]
            print(f"--- Found {len(diff)} difference ---")
            if len(diff) < 1:
                return
            target_files = [split.metadata["source"] for split in diff]
        self.delete_files(target_files)
        print(f"--- Updating files of: {target_files}---")
        self._add_to_database(target_files)

    def _query_files(self):
        files = os.listdir(self.filedir)
        for file in files:
            self._files.append(self.filedir + file)
        print(f"--- Files queried: {self._files} ---")

    def _existing_files(self):
        existing_files = list(
            set(
                [
                    metadata["source"]
                    for metadata in self._vectorstore.get()["metadatas"]
                ]
            )
        )
        return existing_files

    def _get_file_extension(self, filename):
        return filename[filename.rindex(".") + 1 : len(filename) + 1]

    def _add_to_database(self, files):
        splits = []
        for file in files:
            splits += self._file_type_mapping[self._get_file_extension(file)](file)
        split_ids = self._generate_ids(splits)
        assert len(split_ids) == len(splits)
        print(f"--- Adding new {len(splits)} splits into vectorstore ---")
        self._vectorstore.add_documents(documents=splits, ids=split_ids)
        print("--- Loading new files complete ---")

    @staticmethod
    def _generate_ids(splits):
        ids = []
        curDoc = splits[0].metadata["source"]
        split_count = 0
        for split in splits:
            if split.metadata["source"] != curDoc:
                split_count = 0
                curDoc = split.metadata["source"]
            ids.append(split.metadata["source"] + f":{split_count}")
            split_count += 1
        return ids

    def _docx_handler(self, file):
        return self._text_spliter.split_documents(Docx2txtLoader(file).load())

    def _xml_handler(self, file):
        return self._text_spliter.split_documents(UnstructuredXMLLoader(file).load())

    def _pdf_handler(self, file):
        return self._text_spliter.split_documents(PyPDFLoader(file).load())

    def _text_handler(self, file):
        return self._text_spliter.split_documents(TextLoader(file).load())
