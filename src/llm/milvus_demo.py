import os

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db.milvus.milvus_client import connect_to_milvus
from llm.llm_api import get_llm_api

data_dir = r"D:\project\python\fastapi\fastApiDemo\src\data\IKEA"
master_table = os.path.join(data_dir, "master_table") + r"\master_table_febrary.csv"
external_drivers = os.path.join(data_dir, "drivers_log") + r"\drivers_log.csv"
furnishing = os.path.join(data_dir, "furnishing") + r"\HFB.csv"


files = [external_drivers, master_table, furnishing]
documents = []
for file in files:
    print(file)
    loader = CSVLoader(file, encoding='utf-8')
    document = loader.load()
    documents.extend(document)

splitter = RecursiveCharacterTextSplitter()

split_documents = splitter.split_documents(documents)


milvus_client = connect_to_milvus()
milvus_client.save_to_vector(split_documents)
country = ["FR"]
query = f"what are the AOVs and sales index of China, Canada and Japan? Then what is the sales growth of workspaces?"

queries = f"""With the given drivers log table containing sales events of {country} in Jan"
        f" follow the steps to conclude the drivers log factors that affect the sales\n,"
        "Step 1: Find out factors that have negative impact on sales\n,"
        "Step 2: Find out factors that have positive impact on sales\n,"
        "Step 3: Based on these factors, conclude the main drivers for sales and provide implication and"
        "recommendations."""



searches = milvus_client.search_similarity(queries, 30)

# for doc in documents:
#     if 'de' in doc.page_content.lower():
#         print(doc.page_content)
#         print("-" * 50)

# for result in searches:
#         print(f"Content: {result.page_content}")
#         print(f"Metadata: {result.metadata}")
#         print("-" * 50)

# model = llm.LlmApi('gpt-4o-mini-2024-07-18', 0.9)
model = get_llm_api()
res = model.RAG_chat(milvus_client, queries, searches)
print(res)
