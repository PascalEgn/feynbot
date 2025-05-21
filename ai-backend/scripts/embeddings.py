from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv

from backend.src.ir_pipeline.utils.embeddings import VLLMOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opensearchpy.helpers import scan
from utils import (
    get_inspire_os_client,
    get_os_query,
    get_vector_os_client,
    process_hit,
)

embeddings = VLLMOpenAIEmbeddings(
    model_name=getenv("EMBEDDING_MODEL"),
    openai_api_base=f"{getenv('API_BASE')}/v1",
    openai_api_key=getenv("KUBEFLOW_API_KEY"),
    default_headers=(
        {"Host": getenv("KUBEFLOW_EMBEDDING_HOST")}
        if getenv("KUBEFLOW_EMBEDDING_HOST")
        else {}
    ),
    timeout=60,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


inspire_os_client = get_inspire_os_client()
vector_store = get_vector_os_client(embeddings, index_name="embeddings_bge-m3_nucl-ex")


os_query = get_os_query(arXiv_category="nucl-ex", full_text_available=True)

record_count = inspire_os_client.search(
    body={"query": os_query, "size": 0, "track_total_hits": True}, index="records-hep"
)["hits"]["total"]["value"]
print(f"Found {record_count} records")

results = scan(
    inspire_os_client,
    query={"query": os_query},
    index="records-hep",
    size=1000,
    scroll="2m",
)

count = 0
# for idx, hit in enumerate(results):
#     try:
#         if process_hit(idx, hit, vector_store, True, text_splitter):
#             count += 1
#     except Exception as e:
#         print(f"[{idx}] Unexpected error: {e}")
# print(f"Processed {count}/{record_count} records.")

with ThreadPoolExecutor(max_workers=None) as executor:
    futures = {
        executor.submit(process_hit, idx, hit, vector_store, False, text_splitter): idx
        for idx, hit in enumerate(results)
    }

    for future in as_completed(futures):
        idx = futures[future]
        try:
            if future.result():
                count += 1
        except Exception as e:
            print(f"[{idx}] Unexpected error: {e}")

print(f"Finished processing. Successfully processed {count}/{record_count} records.")
