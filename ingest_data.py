from get_dataset import prepare_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain import FAISS
from langchain_community.vectorstores import FAISS

def main():
    data = prepare_dataset()
    texts = data["text"]
    metadatas = data["metadata"]
    # print(texts[0])
    # print(metadatas[0])

    # EMBEDDING MODEL
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    text_embeddings = embedding_model.embed_documents(texts)

    text_embedding_pairs = list(zip(texts, text_embeddings))

    # VECTOR DB
    vector_db = FAISS.from_embeddings(text_embedding_pairs, embedding_model, metadatas)
    vector_db.save_local("warsztat")

if __name__ == "__main__":
    main()