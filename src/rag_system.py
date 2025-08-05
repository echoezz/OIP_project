import os
import json
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

# PDF processing
import PyPDF2

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Text processing and embeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGSystem:
    """RAG system with Qdrant vector database for pest management knowledge base"""
    
    def __init__(
        self, 
        collection_name: str = "pest_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333",
        knowledge_base_path: str = None
    ):
        """
        Initialize RAG system with Qdrant vector database
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model for embeddings
            qdrant_url: Qdrant server URL
            knowledge_base_path: Path to knowledge base folder
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.qdrant_url = qdrant_url
        
        # Set default knowledge base path
        if knowledge_base_path is None:
            current_dir = Path(__file__).parent
            self.knowledge_base_path = current_dir.parent / "knowledge_base"
        else:
            self.knowledge_base_path = Path(knowledge_base_path)
        
        # Initialize components
        self.embedding_model = None
        self.qdrant_client = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize the system
        self._initialize()
    
    def _initialize(self):
        """Initialize embedding model and Qdrant client"""
        try:
            print("🤖 Initializing RAG system...")
            
            # Load embedding model
            print(f"📥 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Initialize Qdrant client
            print(f"🔗 Connecting to Qdrant at {self.qdrant_url}")
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            
            # Create collection if it doesn't exist
            self._create_collection_if_not_exists()
            
            print("✅ RAG system initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    def _create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"📁 Creating collection: {self.collection_name}")
                
                # Get embedding dimension
                sample_text = "sample text for dimension calculation"
                sample_embedding = self.embedding_model.encode([sample_text])
                vector_size = len(sample_embedding[0])
                
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Collection created with vector size: {vector_size}")
            else:
                print(f"📁 Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += f"\n\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        print(f"⚠️ Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                return text.strip()
                
        except Exception as e:
            print(f"❌ Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def load_json_content(self, json_path: Path) -> str:
        """Load and format JSON content as text"""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Format JSON data as readable text
            if isinstance(data, dict):
                text = f"--- {json_path.name} ---\n"
                text += self._format_dict_as_text(data)
            elif isinstance(data, list):
                text = f"--- {json_path.name} ---\n"
                for i, item in enumerate(data):
                    text += f"\nItem {i + 1}:\n"
                    if isinstance(item, dict):
                        text += self._format_dict_as_text(item)
                    else:
                        text += str(item)
            else:
                text = str(data)
                
            return text
            
        except Exception as e:
            print(f"❌ Failed to load JSON from {json_path}: {e}")
            return ""
    
    def _format_dict_as_text(self, data: dict, level: int = 0) -> str:
        """Format dictionary as readable text"""
        text = ""
        indent = "  " * level
        
        for key, value in data.items():
            if isinstance(value, dict):
                text += f"{indent}{key}:\n"
                text += self._format_dict_as_text(value, level + 1)
            elif isinstance(value, list):
                text += f"{indent}{key}:\n"
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        text += f"{indent}  - Item {i + 1}:\n"
                        text += self._format_dict_as_text(item, level + 2)
                    else:
                        text += f"{indent}  - {item}\n"
            else:
                text += f"{indent}{key}: {value}\n"
        
        return text
    
    def process_knowledge_base(self):
        """Process only the Appendix-4-Pest-Management-Framework.pdf from knowledge base"""
        try:
            print(f"📚 Processing knowledge base at: {self.knowledge_base_path}")
            
            if not self.knowledge_base_path.exists():
                print(f"❌ Knowledge base path does not exist: {self.knowledge_base_path}")
                return
            
            documents = []
            
            # Process only the specific PDF file
            target_pdf = self.knowledge_base_path / "Appendix-4-Pest-Management-Framework.pdf"
            
            if target_pdf.exists():
                print(f"📖 Processing target PDF: {target_pdf.name}")
                text = self.extract_text_from_pdf(target_pdf)
                if text:
                    documents.append({
                        'content': text,
                        'source': str(target_pdf.name),
                        'type': 'pdf'
                    })
                    print(f"✅ Extracted {len(text)} characters from {target_pdf.name}")
                else:
                    print(f"⚠️ No text extracted from {target_pdf.name}")
            else:
                print(f"❌ Target PDF not found: {target_pdf}")
                return
            
            if not documents:
                print("⚠️ No documents found to process")
                return
            
            # Split documents into chunks and create embeddings
            print(f"🔪 Splitting {len(documents)} document into chunks...")
            self._index_documents(documents)
            
        except Exception as e:
            print(f"❌ Failed to process knowledge base: {e}")
            raise
    
    def _index_documents(self, documents: List[Dict[str, Any]]):
        """Split documents into chunks and index them in Qdrant"""
        try:
            all_chunks = []
            
            for doc in documents:
                print(f"🔪 Splitting document: {doc['source']}")
                
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc['content'])
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'content': chunk,
                        'source': doc['source'],
                        'type': doc['type'],
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })
            
            print(f"📦 Created {len(all_chunks)} chunks total")
            
            # Create embeddings in batches
            batch_size = 32
            points = []
            
            print("🔢 Creating embeddings...")
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                texts = [chunk['content'] for chunk in batch]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(texts)
                
                # Create points for Qdrant
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    point_id = str(uuid.uuid4())
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload={
                                'content': chunk['content'],
                                'source': chunk['source'],
                                'type': chunk['type'],
                                'chunk_id': chunk['chunk_id'],
                                'total_chunks': chunk['total_chunks']
                            }
                        )
                    )
                
                print(f"✅ Processed batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
            
            # Upload to Qdrant
            print(f"⬆️ Uploading {len(points)} points to Qdrant...")
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✅ Successfully indexed {len(points)} document chunks!")
            
        except Exception as e:
            print(f"❌ Failed to index documents: {e}")
            raise
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity with enhanced scoring"""
        try:
            print(f"🔍 Searching for: '{query}' (limit: {limit})")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in Qdrant with higher limit to allow for filtering
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit * 2  # Get more results to filter better ones
            )
            
            # Format and enhance results
            results = []
            query_words = set(query.lower().split())
            
            for result in search_results:
                content = result.payload['content']
                
                # Calculate additional relevance scores
                content_words = set(content.lower().split())
                keyword_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
                
                # Boost score for pest management terms
                pest_boost = 0
                pest_terms = ['pest', 'control', 'organic', 'treatment', 'management', 'prevention', 'ipm', 'biological']
                for term in pest_terms:
                    if term in content.lower():
                        pest_boost += 0.05
                
                # Calculate final score combining vector similarity and keyword relevance
                final_score = result.score + (keyword_overlap * 0.2) + pest_boost
                
                results.append({
                    'content': content,
                    'source': result.payload['source'],
                    'type': result.payload['type'],
                    'chunk_id': result.payload['chunk_id'],
                    'score': final_score,
                    'vector_score': result.score,
                    'keyword_relevance': keyword_overlap,
                    'metadata': {
                        'total_chunks': result.payload['total_chunks']
                    }
                })
            
            # Sort by enhanced score and take top results
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:limit]
            
            print(f"✅ Found {len(results)} relevant chunks (enhanced scoring)")
            for i, r in enumerate(results[:3]):  # Log top 3
                print(f"  {i+1}. {r['source']} - Score: {r['score']:.3f} (vector: {r['vector_score']:.3f}, keywords: {r['keyword_relevance']:.3f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for a query, formatted for LLM consumption"""
        try:
            # Search for relevant chunks
            results = self.search(query, limit=10)
            
            if not results:
                return ""
            
            # Build context string
            context_parts = []
            current_length = 0
            
            for result in results:
                content = result['content'].strip()
                source_info = f"[Source: {result['source']}]"
                
                # Estimate length with source info
                part_length = len(content) + len(source_info) + 20  # Extra for formatting
                
                if current_length + part_length > max_context_length:
                    break
                
                context_parts.append(f"{source_info}\n{content}")
                current_length += part_length
            
            if context_parts:
                context = "\n\n---\n\n".join(context_parts)
                print(f"📄 Retrieved context: {len(context)} characters from {len(context_parts)} sources")
                return context
            else:
                return ""
                
        except Exception as e:
            print(f"❌ Failed to get context: {e}")
            return ""
    
    def reset_collection(self):
        """Delete and recreate the collection (useful for reindexing)"""
        try:
            print(f"🗑️ Resetting collection: {self.collection_name}")
            
            # Delete collection if it exists
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                print("✅ Collection deleted")
            except:
                print("ℹ️ Collection didn't exist")
            
            # Recreate collection
            self._create_collection_if_not_exists()
            print("✅ Collection reset complete")
            
        except Exception as e:
            print(f"❌ Failed to reset collection: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': collection_info.points_count,
                'status': collection_info.status,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance
            }
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Process knowledge base
    print("Processing knowledge base...")
    rag.process_knowledge_base()
    
    # Test search
    print("\nTesting search...")
    results = rag.search("organic pest control methods", limit=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.3f}):")
        print(f"Source: {result['source']}")
        print(f"Content: {result['content'][:200]}...")
    
    # Test context retrieval
    print("\nTesting context retrieval...")
    context = rag.get_context_for_query("how to control aphids organically")
    print(f"Context length: {len(context)} characters")
    
    # Collection info
    print("\nCollection info:")
    info = rag.get_collection_info()
    print(json.dumps(info, indent=2))
