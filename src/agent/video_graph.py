from __future__ import annotations

import json

import numpy as np
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from pydantic import BaseModel
from sqlalchemy import create_engine, text

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
tag_llm = OllamaLLM(model="qwen2.5")
engine = create_engine("postgresql+psycopg2://postgres:testing123@localhost:5432/postgres")

TAG_PROMPT = PromptTemplate(
    template="""
Analyze this video content and generate 8-12 relevant tags.

Video Details: {details}

Generate tags that are:
- Specific and searchable
- Relevant to the content
- Mix of broad and specific terms
- Good for categorization

Return ONLY a JSON array of tags like this:
["tag1", "tag2", "tag3", "tag4", "tag5"]

JSON Response:
""",
    input_variables=["details"]
)


class State(BaseModel):
    url: str = ""
    video_details: str = ""
    tags: list[str] = []
    embedding: list = None


async def download_video_node(state: State):
    url = state.url
    loader = YoutubeLoaderDL.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    state.url = url
    state.video_details = documents[0].metadata.get("description")
    return state


async def generator_tags_node(state: State):
    try:
        if not state.video_details:
            print("⚠ No content for tag generation")
            state.tags = []
            return state

        # Limit content length
        details = state.video_details[:3000]

        # Generate prompt
        prompt = TAG_PROMPT.format(details=details)

        print("🏷️ Generating tags...")

        # Get LLM response
        response = tag_llm.invoke(prompt)

        # Parse JSON response
        tags = json.loads(response)
        print(f"<UNK> Tags generated: {tags}")
        state.tags = tags

    except Exception as ex:
        print(ex)

    return state


async def embed_node(state: State):
    state.embedding = embedding_model.embed_query(state.video_details)
    return state

async def store_node(state: State):
    with engine.connect() as conn:
        insert_stmt = text("""
            INSERT INTO videos (url, details, embedding, tags)
            VALUES (:url, :details, :embedding, :tags)
        """)
        conn.execute(insert_stmt, {
            "url": state.url,
            "details": state.video_details,
            "embedding": state.embedding,
            "tags": state.tags,
        })
        conn.commit()
    return state


graph = StateGraph(State)

graph.add_node("video_downloader", download_video_node)
graph.add_node("tag_generator", generator_tags_node)
graph.add_node("embedding", embed_node)
graph.add_node("storing", store_node)

graph.add_edge(START, "video_downloader")
graph.add_edge("video_downloader", "tag_generator")
graph.add_edge("tag_generator", "embedding")
graph.add_edge("embedding", "storing")
graph.add_edge("storing", END)

final_graph = graph.compile()
