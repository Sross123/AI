from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embadding_model = MistralAIEmbeddings(model="mistral-embed")

vectorStore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embadding_model
)

retriever = vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k": 10,
        "lambda_mult":0.5
    }
)

llm = ChatMistralAI(model_name="mistral-small-2603")

#prompt template
template = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are a helpful AI Assistant who can friendly explain.
         Use ONLY the provided context to answer the question.
         Do not make up facts. Do not use outside knowledge.
         
         If the answer is not provided in the context,
         say: "I could not find the answer in the document."
         
         CRITICAL RULE: Before giving the final answer, check if every fact you stated is directly present in the context. If not, delete that fact.
         """),
        ("human", """
         Context: {context}
         
         Question: {question}
         """)
    ]
)

print("Rag system created")
print("Press 0 to exit")

# ---- BASIC GREETING ADDED HERE ----
print("\n🤖 AI: Hello Shashi! How can I help you with your document today?") 
# -----------------------------------

while True:
    query = input("\nYou: ")
    if query == "0":
        # ---- EXIT MESSAGE ----
        print("🤖 AI: Goodbye! Have a great day!")
        # ----------------------
        break
    
    # Get relevant chunks
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Invoke prompt with BOTH context and question
    final_prompt = template.invoke({
        "context": context,
        "question": query
    })
    
    # Get response from LLM and print it
    response = llm.invoke(final_prompt)
    print(f"\nAI: {response.content}")