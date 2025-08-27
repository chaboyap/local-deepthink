# app/rag/raptor.py

import asyncio
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from sklearn.cluster import KMeans
import numpy as np

class RAPTORRetriever(BaseRetriever):
    """A simple retriever that wraps the RAPTOR index."""
    raptor_index: Any
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.raptor_index.retrieve(query)

class RAPTOR:
    """
    Implementation of the RAPTOR (Recursive Abstractive Processing for
    Tree-Organized Retrieval) method for building a hierarchical RAG index.
    """
    def __init__(self, llm, embeddings_model, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tree = {}
        self.all_nodes: Dict[str, Document] = {}
        self.vector_store = None

    async def add_documents(self, documents: List[Document]):
        """
        Builds the RAPTOR index from a list of documents.
        """
        logging.info("Step 1: Assigning IDs to initial chunks (Level 0)...")
        level_0_node_ids = []
        for i, doc in enumerate(documents):
            node_id = f"0_{i}"
            self.all_nodes[node_id] = doc
            level_0_node_ids.append(node_id)
        self.tree[str(0)] = level_0_node_ids
        
        current_level = 0
        while len(self.tree.get(str(current_level), [])) > 1:
            next_level = current_level + 1
            logging.info(f"Step 2: Building Level {next_level} of the tree...")
            current_level_node_ids = self.tree[str(current_level)]
            current_level_docs = [self.all_nodes[nid] for nid in current_level_node_ids]
            clustered_indices = await self._cluster_nodes(current_level_docs)
            
            next_level_node_ids = []
            logging.info(f"Summarizing Level {next_level}...")
            
            summarization_tasks = []
            for i, indices in enumerate(clustered_indices):
                if not indices: continue
                cluster_docs = [current_level_docs[j] for j in indices]
                summarization_tasks.append(self._summarize_cluster(cluster_docs, next_level, i))
            
            summaries = await asyncio.gather(*summarization_tasks)

            for i, summary_tuple in enumerate(summaries):
                 if summary_tuple is None: continue
                 summary_doc, _ = summary_tuple
                 if summary_doc is None: continue

                 node_id = f"{next_level}_{i}"
                 self.all_nodes[node_id] = summary_doc
                 next_level_node_ids.append(node_id)

            if not next_level_node_ids:
                logging.warning(f"WARNING: Could not generate any valid summaries for Level {next_level}. Halting tree construction.")
                break

            self.tree[str(next_level)] = next_level_node_ids
            current_level = next_level

        logging.info("Step 3: Creating final vector store from all nodes...")
        final_docs = list(self.all_nodes.values())
        
        # Use asyncio.to_thread for the blocking FAISS call
        self.vector_store = await asyncio.to_thread(
            FAISS.from_documents, documents=final_docs, embedding=self.embeddings_model
        )
        logging.info("RAPTOR index built successfully!")

    async def _cluster_nodes(self, docs: List[Document]) -> List[List[int]]:
        num_docs = len(docs)
        if num_docs == 0:
            return []
        if num_docs <= 5:
            logging.info(f"Grouping {num_docs} remaining nodes into a single summary to finalize the tree.")
            return [list(range(num_docs))]

        logging.info(f"Embedding {num_docs} nodes for clustering...")
        embeddings = await self.embeddings_model.aembed_documents([doc.page_content for doc in docs])
        
        n_clusters = max(2, num_docs // 5)
        if n_clusters >= num_docs:
            n_clusters = num_docs - 1 if num_docs > 1 else 1

        if n_clusters <= 1:
            return [list(range(num_docs))]
        
        logging.info(f"Clustering {num_docs} nodes into {n_clusters} groups...")
        
        kmeans = await asyncio.to_thread(
            KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit, 
            np.array(embeddings)
        )
        
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        return clusters

    async def _summarize_cluster(self, cluster_docs: List[Document], level: int, cluster_index: int) -> Tuple[Document, dict] | None:
        """Summarizes a cluster of documents using the LLM."""
        context = "\n\n---\n\n".join([doc.page_content for doc in cluster_docs])
        prompt_template = (
            "You are a focused AI assistant that synthesizes information. "
            "Your sole task is to create a dense, abstractive summary of the following text content, which contains outputs from multiple AI agents. "
            "Focus on extracting the core ideas, proposed solutions, and key reasoning steps. "
            "Do not comment on the process or the quality of the text. Simply summarize it. "
            "Text to summarize:\n\n"
            "---BEGIN CONTEXT---\n"
            "{context}\n"
            "---END CONTEXT---\n\n"
            "Provide your summary:"
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"context": context})
            summary = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logging.error(f"Summarizer LLM call failed for cluster {cluster_index + 1} at Level {level}: {e}")
            return None
            
        MIN_SUMMARY_LENGTH = 50 
        if len(summary) < MIN_SUMMARY_LENGTH:
            logging.warning(f"Summarizer LLM produced a very short (likely failed) summary for cluster {cluster_index + 1}. Discarding.")
            return None

        source_agent_ids = sorted(list(set(
            doc.metadata.get("agent_id") for doc in cluster_docs if "agent_id" in doc.metadata
        )))
        
        all_summarized_ids = []
        for doc in cluster_docs:
            if "summary_of" in doc.metadata and doc.metadata["summary_of"] is not None:
                all_summarized_ids.extend(doc.metadata["summary_of"])
        combined_source_ids = sorted(list(set(source_agent_ids + all_summarized_ids)))

        combined_metadata = { "level": level, "summary_of": combined_source_ids }
        summary_doc = Document(page_content=summary, metadata=combined_metadata)
        
        logging.info(f"Summarized cluster {cluster_index + 1} for Level {level}...")
        return summary_doc, combined_metadata
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieves relevant documents from the vector store."""
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []
    
    def as_retriever(self) -> BaseRetriever:
        """Returns a LangChain-compatible retriever instance."""
        return RAPTORRetriever(raptor_index=self)