from __future__ import annotations

from langgraph.graph import StateGraph, END
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from langchain_ollama import OllamaEmbeddings


embedding_model = OllamaEmbeddings(model="nomic-embed-text")


class State:
    video_details: str = ""
    embedding: list = None


async def download_video_node(state: State):
    loader = YoutubeLoaderDL.from_youtube_url(
        "https://www.youtube.com/watch?v=SvMabklWzDY", add_video_info=True
    )
    documents = loader.load()
    state.video_details = documents[0].metadata
    return state


async def generator_tags_node(state: State):
    state.video_details = "generate tags"
    return state

async def embed_node(state: State):
    state.embedding = embedding_model.embed_query(state.video_details)
    return state

async def store_node(state: State):
    state.video_details = "store video"
    return state


graph = StateGraph(State)

graph.add_node("video_downloader", download_video_node)
graph.add_node("tag_generator", generator_tags_node)
graph.add_node("embedding", embed_node)
graph.add_node("storing", store_node)

graph.set_entry_point("video_downloader")
graph.add_edge("video_downloader", "tag_generator")
graph.add_edge("tag_generator", "embedding")
graph.add_edge("embedding", "storing")
graph.add_edge("storing", END)

final_graph = graph.compile(name="Video Processor Graph")
