from langchain_ollama import OllamaEmbeddings
from sqlalchemy import create_engine, text

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

engine = create_engine("postgresql+psycopg2://postgres:testing123@localhost:5432/postgres")

def search_videos(query: str, top_k: int = 5):
    query_embedding = embedding_model.embed_query(query)

    sql = text("""
               SELECT id, url, details
               FROM videos
               ORDER BY embedding <-> :embedding::vector
        LIMIT :limit
               """)

    with engine.connect() as conn:
        result = conn.execute(sql, {"embedding": query_embedding, "limit": top_k})
        videos = result.fetchall()

    results = []
    for row in videos:
        results.append({
            "id": row.id,
            "url": row.url,
            "details": row.details
        })
    return results

if __name__ == "__main__":
    query_text = "any Billie's songs"
    results = search_videos(query_text)

    for video in results:
        print(f"URL: {video['url']}")
        print(f"Details: {video['details'][:100]}...")
        print("------")
