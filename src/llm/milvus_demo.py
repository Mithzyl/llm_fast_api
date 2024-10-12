import os

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db.milvus.milvus_client import connect_to_milvus
from llm.llm_api import get_llm_api

data_dir = r"D:\project\python\fastapi\fastApiDemo\src\data\IKEA"
master_table = os.path.join(data_dir, "master_table") + r"\master_table_febrary.csv"
external_drivers = os.path.join(data_dir, "drivers_log") + r"\drivers_log.csv"
furnishing = os.path.join(data_dir, "furnishing") + r"\HFB.csv"


files = [external_drivers, master_table]
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
country = ["Canada", "China", "Japan"]
query = f"what are the AOVs and sales index of China, Canada and Japan? Then what is the sales growth of workspaces?"

queries = f"""
            f"There is a master table containing sales business data of a month in a country which"
           f" performs worse or better than last year, "
           f"check what columns values may be contributed to the decline,"
           f"the goal is to analyze the business performance of {country}
           f"Step 1: check conversion rate, it refers to the percentages of shopping visitors buying items,\n"
           f"compare the conversion rate with the global conversion rate, if the conversion exceeds,"
           f" it means the conversion rate is high, factors driving the conversion rate can be investigated.\n"
           f"Step 2: check returns and cancellations rate, it refers to the percentages of shopping visitors"
           f" return or cancel orders, compare the returns and cancellations rate with the global"
           f" returns and cancellations rate, if the conversion exceeds,"
           f" it means the conversion returns and cancellations rate is high,"
           f" customers are returning items than expected. Factors driving the conversion rate should be investigated.\n"
           f" Step 3: check the SSC Diversions Truck Returns column value,"
           f" SSC Diversions Truck Returns means The SS containment is one of the RCMP's main goals and"
           f" is used to evaluate how well the customers are diverting to use the self-service"
           f" capabilities instead of contacting the CSC through manual channels (phone, chat, email and SoMe)."
           f" Compare with the average SSC Diversions Truck Returns.\n"
           f" Step 4: check Effort CSAT score, CSAT measures overall customer satisfaction, effort CSAT"
           f" score means a score of 4 or 5 out of 5 on the question 'The level of effort to get an"
           f" answer or solution' on the survey of customer satisfaction after"
           f" talking with a customer support agent. effort CSAT has a negative impact on sales performance. \n"
           f" Compare the Effort CSAT score with the global Effort CSAT score, if the Effort CSAT score exceeds,"
           f" it means the Effort CSAT score is high, factors driving the Effort CSAT score can be investigated.\n"
           f" Step 5: check Competent CSAT score, CSAT measures overall customer satisfaction, competent CSAT"
           f" score means a score of 4 or 5 out of 5 on the question  'The agent's knowledge'"
           f" on the survey of customer satisfaction after"
           f" talking with a customer support agent. Competent CSAT has a negative impact on sales performance.\n"
           f" Compare the Competent CSAT score with the global Competent CSAT score, if the Competent CSAT score exceeds,"
           f" it means the Competent CSAT score is high, factors driving the Competent CSAT score can be investigated.\n"
           f" Step 6: check Friendly CSAT score, CSAT measures overall customer satisfaction, friendly CSAT"
           f" score means a score of 4 or 5 out of 5 on the question 'The agent's attitude' on the"
           f" survey of customer satisfaction after"
           f" talking with a customer support agent. Friendly CSAT has a negative impact on sales performance.\n"
           f" Compare the Friendly CSAT score with the global Friendly CSAT score, if the Friendly CSAT score exceeds,"
           f" it means the Friendly CSAT score is high, factors driving the Friendly CSAT score can be investigated.\n"
           f" Step 7: check Sales Index LY B2B Remote value, index above 100 means incline while below 100 indicates decline\n"
           f" Then form a implication and recommendations on how to reach sales goal"
            """


searches = milvus_client.search_similarity(queries, 10)

# for doc in documents:
#     if 'de' in doc.page_content.lower():
#         print(doc.page_content)
#         print("-" * 50)

for result in searches:
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
        print("-" * 50)

# model = llm.LlmApi('gpt-4o-mini-2024-07-18', 0.9)
model = get_llm_api()
res = model.RAG_chat(milvus_client, query)
print(res)
