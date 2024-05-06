from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, MessagesPlaceholder
from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import logging
from database import DB_Handler

logging.getLogger('chromadb').setLevel(logging.WARNING)

class Model:
    def __init__(self, temperature=None):
        self.model = Ollama(model="llama3:8b", temperature=temperature) 
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True, max_length=5)
        self.store = {}

    def query_with_memory(self, query_text: str) -> str:
        self.memory.chat_memory.add_user_message(query_text)

        system_prompt = """Given a chat history and the latest user question \
                        which might reference context in the chat history, formulate a standalone question \
                        which can be understood without the chat history. Do NOT answer the question, \
                        just reformulate it if needed and otherwise return it as is."""

        contextualized_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        db = DB_Handler().get_db()

        retriever = db.as_retriever(
            search_type="mmr",
            # {'filter': {'paper_title':'GPT-4 Technical Report'}}
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        retriever_with_memory = create_history_aware_retriever(
            self.model, retriever, contextualized_prompt
        )

        context_prompt = """You are a smart and witty AI Assistant helping a human, \
                            use following retrieved context in order to answer the human's question, \
                            if you don't know the answer then just say that you don't know. \
                            
                            {context}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", context_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)

        chain = create_retrieval_chain(
            retriever_with_memory,
            question_answer_chain
        )

        response = chain.invoke({"input": query_text, "history": self.memory.load_memory_variables({})['history']})
        self.memory.save_context(inputs={"input": query_text}, outputs={"output": str(response["answer"])})

        return response["answer"]
