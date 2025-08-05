import json
import os
import re
import requests
from typing import Dict, List, Tuple, Optional
import random

from .rag_system import RAGSystem

class RAGChatBot:
    """Enhanced chatbot with RAG capabilities for pest management"""
    
    def __init__(
        self, 
        ollama_model="llama3.2:3b", 
        ollama_url="http://localhost:11434",
        use_rag=True,
        qdrant_url="http://localhost:6333"
    ):
        """Initialize RAG-enhanced chatbot for organic pest management"""
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.use_rag = use_rag
        self.conversation_history = []
        
        # Initialize RAG system if enabled
        self.rag_system = None
        if self.use_rag:
            try:
                print("🤖 Initializing RAG system...")
                self.rag_system = RAGSystem(qdrant_url=qdrant_url)
                print("✅ RAG system initialized")
            except Exception as e:
                print(f"⚠️ RAG system initialization failed: {e}")
                print("📝 Falling back to basic chatbot mode")
                self.use_rag = False
        
        # Load fallback knowledge
        self.pest_database = self.load_pest_database()
        
        # Enhanced system prompt with RAG context
        self.system_prompt = """You are an expert organic pest management consultant with access to comprehensive pest management documentation and research. You have deep knowledge of:

- Pest identification and biology
- Organic and natural treatment methods  
- Integrated Pest Management (IPM) strategies
- Prevention and monitoring techniques
- Beneficial insects and biological controls
- Companion planting and cultural practices
- Sustainable gardening practices
- Timing and application methods

When provided with context from research documents, prioritize that information in your responses while combining it with your expertise.

Always provide:
1. Safe, organic solutions first
2. Clear, step-by-step actionable advice
3. Prevention tips and monitoring strategies
4. Environmental and safety considerations
5. Timing recommendations and seasonal guidance
6. Multiple treatment options when available

Keep responses helpful, practical, and focused on organic/natural methods. Avoid recommending synthetic pesticides unless specifically asked about integrated approaches.

If context is provided from documents, cite the source information naturally in your response.
"""
    
    def load_pest_database(self):
        """Load pest database from JSON file"""
        try:
            knowledge_base_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')
            pest_db_path = os.path.join(knowledge_base_path, 'pest_database.json')
            
            if os.path.exists(pest_db_path):
                with open(pest_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load pest database: {e}")
        
        return {}
    
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Ollama connection failed: {e}")
            return False
    
    def check_rag_system(self):
        """Check if RAG system is available and functional"""
        if not self.use_rag or not self.rag_system:
            return False
        
        try:
            # Test with a simple query
            results = self.rag_system.search("test", limit=1)
            return True
        except Exception as e:
            print(f"⚠️ RAG system check failed: {e}")
            return False
    
    def get_rag_context(self, query: str) -> str:
        """Get relevant context from RAG system with summarization"""
        if not self.use_rag or not self.rag_system:
            return ""
        
        try:
            # Get the top relevant chunks
            results = self.rag_system.search(query, limit=3)
            
            if not results:
                print("📚 No relevant RAG context found")
                return ""
            
            # Check if the best result meets the relevance threshold
            best_score = results[0]['score'] if results else 0
            if best_score < 0.65:
                print(f"📊 Best relevance score {best_score:.3f} is below threshold (0.65). Using Ollama without RAG context.")
                return ""
            
            # Filter results to only include those above threshold
            relevant_results = [r for r in results if r['score'] >= 0.65]
            
            if not relevant_results:
                print("📊 No results above relevance threshold (0.65). Using Ollama without RAG context.")
                return ""
            
            # Summarize and format the most relevant chunks
            context_parts = []
            total_chars = 0
            max_context_length = 1500  # Reduced for more focused responses
            
            for i, result in enumerate(relevant_results):
                if total_chars >= max_context_length:
                    break
                
                content = result['content'].strip()
                source = result['source']
                score = result['score']
                
                # Summarize long chunks (keep most relevant sentences)
                summarized_content = self._summarize_chunk(content, query, max_length=400)
                
                # Format with relevance score
                chunk_info = f"**Source: {source}** (Relevance: {score:.1%})\n{summarized_content}"
                
                context_parts.append(chunk_info)
                total_chars += len(chunk_info)
                
                print(f"📄 Chunk {i+1}: {len(summarized_content)} chars from {source} (score: {score:.3f}) ✅")
            
            if context_parts:
                context = "\n\n---\n\n".join(context_parts)
                print(f"📚 Retrieved {len(context_parts)} high-relevance chunks: {len(context)} total characters")
                return f"Relevant information from pest management resources:\n\n{context}\n\n"
            else:
                return ""
                
        except Exception as e:
            print(f"⚠️ Failed to get RAG context: {e}")
            return ""
    
    def _summarize_chunk(self, content: str, query: str, max_length: int = 400) -> str:
        """Summarize a content chunk to focus on query-relevant information"""
        try:
            # Split into sentences
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            
            if len(content) <= max_length:
                return content
            
            # Score sentences based on query relevance
            query_words = set(query.lower().split())
            scored_sentences = []
            
            for sentence in sentences:
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                sentence_words = set(sentence.lower().split())
                # Simple relevance scoring based on word overlap
                overlap = len(query_words.intersection(sentence_words))
                score = overlap / len(query_words) if query_words else 0
                
                # Boost score for pest-related terms
                pest_terms = ['pest', 'control', 'organic', 'treatment', 'management', 'prevention']
                for term in pest_terms:
                    if term in sentence.lower():
                        score += 0.1
                
                scored_sentences.append((sentence, score))
            
            # Sort by relevance and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Build summary with most relevant sentences
            summary_parts = []
            current_length = 0
            
            for sentence, score in scored_sentences:
                if current_length + len(sentence) > max_length:
                    break
                summary_parts.append(sentence)
                current_length += len(sentence)
                
                # Stop if we have enough content
                if len(summary_parts) >= 3 and current_length > max_length * 0.7:
                    break
            
            if summary_parts:
                summary = '. '.join(summary_parts)
                if not summary.endswith('.'):
                    summary += '.'
                return summary
            else:
                # Fall back to truncated original if summarization fails
                return content[:max_length] + "..." if len(content) > max_length else content
                
        except Exception as e:
            print(f"⚠️ Chunk summarization failed: {e}")
            # Fall back to truncated original
            return content[:max_length] + "..." if len(content) > max_length else content
    
    def call_ollama(self, prompt: str, context: str = "") -> str:
        """Call Ollama API with the given prompt and context"""
        try:
            print(f"🤖 Calling Ollama with prompt: {prompt[:100]}...")
            
            # Build full prompt with context
            if context:
                full_prompt = f"Context from pest management resources:\n{context}\n\nUser question: {prompt}\n\nAs an organic pest management expert, provide helpful advice based on the context and your expertise:"
            else:
                full_prompt = f"User question: {prompt}\n\nAs an organic pest management expert, provide helpful advice:"
            
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            print(f"📤 Sending request to Ollama...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Increased timeout to 2 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                print(f"✅ Ollama response received: {len(answer)} characters")
                return answer
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return ""
                
        except requests.exceptions.Timeout:
            print("⏰ Ollama request timed out")
            return ""
        except Exception as e:
            print(f"❌ Ollama call failed: {e}")
            return ""
    
    def get_fallback_response(self, user_input: str) -> str:
        """Generate fallback response when Ollama is unavailable"""
        user_lower = user_input.lower()
        
        # Enhanced fallback responses with pest database integration
        if any(word in user_lower for word in ['aphid', 'aphids']):
            return """🐛 **Aphid Control (Organic Methods):**

**Immediate Actions:**
• Spray with water to dislodge aphids
• Apply insecticidal soap spray (2 tbsp per quart water)
• Use neem oil spray in evening

**Natural Predators:**
• Attract ladybugs, lacewings, and parasitic wasps
• Plant companion plants: marigolds, catnip, garlic

**Prevention:**
• Avoid over-fertilizing with nitrogen
• Monitor plants regularly, especially new growth
• Remove weeds that can harbor aphids

*For specific species and advanced techniques, I'd recommend consulting detailed pest management resources.*"""
        
        elif any(word in user_lower for word in ['caterpillar', 'worm', 'larvae']):
            return """🐛 **Caterpillar Management (Organic Approach):**

**Physical Control:**
• Hand-picking in early morning or evening
• Use row covers during egg-laying periods
• Install pheromone traps for monitoring

**Biological Control:**
• Bacillus thuringiensis (Bt) spray for leaf-eating caterpillars
• Encourage birds with bird houses and water sources
• Plant flowers to attract parasitic wasps

**Cultural Practices:**
• Rotate crops annually
• Till soil in fall to expose overwintering pupae
• Remove plant debris and weeds

*The specific approach depends on the caterpillar species and affected plants.*"""
        
        elif any(word in user_lower for word in ['spider mite', 'mite', 'mites']):
            return """🕷️ **Spider Mite Control (Organic Methods):**

**Environmental Control:**
• Increase humidity around plants
• Ensure adequate watering (mites prefer dry conditions)
• Improve air circulation

**Organic Treatments:**
• Spray with water to dislodge mites
• Apply neem oil or horticultural oil
• Use predatory mites (biological control)

**Prevention:**
• Avoid dusty conditions
• Don't over-fertilize with nitrogen
• Monitor regularly with hand lens

*Spider mites can develop resistance quickly, so rotate treatments.*"""
        
        elif any(word in user_lower for word in ['slug', 'snail', 'slugs', 'snails']):
            return """🐌 **Slug and Snail Management:**

**Physical Barriers:**
• Copper strips around plants
• Diatomaceous earth (food grade)
• Crushed eggshells or coffee grounds

**Trapping:**
• Beer traps (shallow dishes)
• Board traps (check and remove daily)
• Citrus rind traps

**Natural Predators:**
• Encourage ground beetles, birds
• Create habitat for beneficial insects

**Cultural Control:**
• Water in morning, not evening
• Remove hiding places (debris, weeds)
• Use drip irrigation instead of overhead watering

*Most active at night and in humid conditions.*"""
        
        elif any(word in user_lower for word in ['prevention', 'prevent', 'avoiding']):
            return """🛡️ **Integrated Pest Prevention Strategies:**

**Cultural Practices:**
• Crop rotation (2-3 year cycles)
• Proper plant spacing for air circulation
• Healthy soil through composting
• Choose resistant varieties when available

**Monitoring:**
• Weekly plant inspections
• Use yellow sticky traps for early detection
• Keep garden journal for tracking patterns

**Beneficial Habitat:**
• Plant diverse flowers to attract beneficial insects
• Provide water sources and shelter
• Avoid broad-spectrum pesticides

**Timing:**
• Plant at optimal times for your region
• Time plantings to avoid peak pest periods
• Harvest promptly to reduce pest attraction

*Prevention is always more effective and economical than treatment.*"""
        
        elif any(word in user_lower for word in ['companion', 'planting', 'plants']):
            return """🌱 **Companion Planting for Pest Management:**

**Repellent Plants:**
• Marigolds - repel nematodes, aphids
• Basil - repels flies, mosquitoes, thrips
• Catnip - stronger than DEET for mosquitoes
• Garlic/Onions - repel many pests

**Trap Crops:**
• Nasturtiums for aphids and cucumber beetles
• Radishes for flea beetles
• Sunflowers for stink bugs

**Beneficial Attractors:**
• Yarrow - attracts predatory wasps
• Dill - attracts beneficial insects
• Sweet alyssum - attracts hover flies

**Polyculture Benefits:**
• Reduces pest buildup
• Confuses pest location
• Provides habitat diversity

*Plan your garden layout to maximize these natural partnerships.*"""
        
        elif any(word in user_lower for word in ['organic', 'natural', 'safe']):
            return """🌿 **Safe Organic Pest Control Principles:**

**Least Toxic First:**
1. Physical removal and barriers
2. Biological controls (beneficial insects)
3. Botanical and mineral-based sprays
4. Targeted organic pesticides as last resort

**Application Safety:**
• Read all labels carefully
• Apply during cooler parts of day
• Avoid treating during bloom when pollinators active
• Wear appropriate protective equipment

**Environmental Considerations:**
• Protect beneficial insects and pollinators
• Avoid contaminating water sources
• Consider impact on soil organisms
• Test treatments on small areas first

**Integrated Approach:**
• Combine multiple strategies
• Focus on long-term ecosystem health
• Monitor and adjust methods based on results

*Even organic treatments can have impacts - use responsibly.*"""
        
        else:
            # Generic helpful response
            responses = [
                """🌱 **General Organic Pest Management Guidance:**

I'd be happy to help with your pest management question! For the most accurate and detailed advice, I recommend:

**Immediate Steps:**
• Identify the specific pest (photos help)
• Assess the extent of damage
• Check for beneficial insects before treating

**Organic Approach:**
• Start with least invasive methods
• Consider the pest's life cycle
• Monitor effectiveness and adjust

**Common Solutions:**
• Insecticidal soap for soft-bodied pests
• Neem oil for various pests and diseases
• Beneficial insects for long-term control
• Physical barriers and traps

*Could you provide more specific details about the pest or problem you're dealing with?*""",

                """🔍 **Pest Identification and Management:**

Effective pest management starts with proper identification. Here's how to proceed:

**Gather Information:**
• What plant is affected?
• Describe the damage you're seeing
• When did you first notice the problem?
• What's the current weather/season?

**Documentation:**
• Take clear photos of pests and damage
• Note patterns (time of day, weather conditions)
• Track which plants are affected

**Research:**
• Consult extension service resources
• Use plant identification apps
• Connect with local gardening groups

**Treatment Selection:**
• Match treatment to specific pest
• Consider beneficial insects in area
• Start with gentlest effective method

*Feel free to share more details about your specific situation!*"""
            ]
            return random.choice(responses)
    
    def respond(self, user_input: str) -> str:
        """Generate response using RAG-enhanced context when possible"""
        try:
            print(f"💬 Processing user input: {user_input[:100]}...")
            
            # Get RAG context if available
            rag_context = ""
            if self.use_rag and self.rag_system:
                rag_context = self.get_rag_context(user_input)
                if not rag_context:
                    print("🤖 No high-relevance RAG context found. Using Ollama with general knowledge.")
            
            # Try Ollama first
            if self.check_ollama_connection():
                response = self.call_ollama(user_input, rag_context)
                if response:
                    # Add conversation to history
                    self.conversation_history.append({
                        'user': user_input,
                        'bot': response,
                        'used_rag': bool(rag_context),
                        'context_length': len(rag_context) if rag_context else 0
                    })
                    return response
            
            # Fallback to pre-defined responses
            print("🔄 Using fallback response system")
            fallback_response = self.get_fallback_response(user_input)
            
            # Enhance fallback with RAG context if available (only high-relevance)
            if rag_context:
                enhanced_response = f"*Based on available pest management resources:*\n\n{rag_context}\n\n---\n\n{fallback_response}"
                self.conversation_history.append({
                    'user': user_input,
                    'bot': enhanced_response,
                    'used_rag': True,
                    'context_length': len(rag_context),
                    'fallback': True
                })
                return enhanced_response
            else:
                self.conversation_history.append({
                    'user': user_input,
                    'bot': fallback_response,
                    'used_rag': False,
                    'context_length': 0,
                    'fallback': True
                })
                return fallback_response
                
        except Exception as e:
            print(f"❌ Error in respond method: {e}")
            return "⚠️ I'm experiencing technical difficulties. Please try again or rephrase your question."
    
    def get_system_status(self) -> Dict[str, any]:
        """Get status of all system components"""
        status = {
            'ollama_connected': self.check_ollama_connection(),
            'rag_enabled': self.use_rag,
            'rag_functional': self.check_rag_system(),
            'conversation_history_length': len(self.conversation_history),
            'pest_database_loaded': bool(self.pest_database)
        }
        
        if self.use_rag and self.rag_system:
            try:
                collection_info = self.rag_system.get_collection_info()
                status['rag_collection_info'] = collection_info
            except:
                status['rag_collection_info'] = None
        
        return status
    
    def initialize_knowledge_base(self):
        """Initialize/reindex the RAG knowledge base (resets existing data)"""
        if not self.use_rag or not self.rag_system:
            return False, "RAG system not available"
        
        try:
            print("🔄 Initializing knowledge base...")
            
            # Reset collection to clear old data
            print("🗑️ Clearing existing vector data...")
            self.rag_system.reset_collection()
            
            # Process only the PDF file
            print("📚 Processing knowledge base...")
            self.rag_system.process_knowledge_base()
            
            return True, "Knowledge base reset and reinitialized successfully with PDF-only content"
        except Exception as e:
            return False, f"Failed to initialize knowledge base: {e}"
    
    def reset_vector_database(self):
        """Reset the vector database to clear all existing data"""
        if not self.use_rag or not self.rag_system:
            return False, "RAG system not available"
        
        try:
            print("🗑️ Resetting vector database...")
            self.rag_system.reset_collection()
            return True, "Vector database reset successfully - ready for reindexing"
        except Exception as e:
            return False, f"Failed to reset vector database: {e}"

# Example usage
if __name__ == "__main__":
    # Initialize RAG chatbot
    chatbot = RAGChatBot(use_rag=True)
    
    # Check system status
    status = chatbot.get_system_status()
    print("System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Initialize knowledge base if RAG is available
    if chatbot.use_rag:
        success, message = chatbot.initialize_knowledge_base()
        print(f"\nKnowledge base initialization: {message}")
    
    # Test conversation
    print("\n" + "="*50)
    print("Testing RAG-enhanced chatbot:")
    print("="*50)
    
    test_queries = [
        "How do I control aphids organically?",
        "What are the best companion plants for pest control?",
        "How can I prevent garden pests naturally?"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = chatbot.respond(query)
        print(f"Bot: {response[:300]}...")
