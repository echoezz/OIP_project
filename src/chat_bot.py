import json
import os
import re
import requests
from typing import Dict, List, Tuple
import random

class OrganicPestChatBot:
    def __init__(self, ollama_model="llama3.2:3b", ollama_url="http://localhost:11434"):
        """Initialize Ollama-powered chatbot for organic pest management"""
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.conversation_history = []
        self.pest_database = self.load_pest_database()
        
        # System prompt for pest management expertise
        self.system_prompt = """You are an expert organic pest management consultant with deep knowledge of:
- Pest identification and biology
- Organic and natural treatment methods
- Prevention strategies
- Companion planting
- Beneficial insects
- Integrated Pest Management (IPM)
- Sustainable gardening practices

Always provide:
1. Safe, organic solutions first
2. Clear, actionable advice
3. Prevention tips when relevant
4. Environmental considerations
5. Timing recommendations

Keep responses helpful, practical, and focused on organic/natural methods. Avoid recommending synthetic pesticides.
"""
    
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Ollama connection failed: {e}")
            return False
    
    def call_ollama(self, prompt: str, context: str = "") -> str:
        """Call Ollama API with the given prompt"""
        try:
            # Simplified prompt construction
            if context:
                full_prompt = f"{context}\n\nUser question: {prompt}\n\nAs an organic pest management expert, provide helpful advice:"
            else:
                full_prompt = f"User question: {prompt}\n\nAs an organic pest management expert, provide helpful advice:"
            
            print(f"🔄 Sending request to Ollama with {len(full_prompt)} characters...")
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 300
                    }
                },
                timeout=120  # Increased timeout for model loading
            )
            
            print(f"📡 Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "I'm sorry, I couldn't generate a response.")
                print(f"✅ Received response: {len(response_text)} characters")
                return response_text
            else:
                error_msg = f"Error: Ollama API returned status {response.status_code}"
                print(f"❌ {error_msg}")
                return error_msg
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama API error: {e}")
            return self.get_fallback_response(prompt)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return self.get_fallback_response(prompt)
    
    def get_fallback_response(self, prompt: str) -> str:
        """Provide fallback responses when Ollama is not available"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["aphid", "aphids"]):
            return """🐛 **Aphid Management:**

**Organic Treatments:**
• Spray with insecticidal soap or neem oil
• Use strong water spray to dislodge aphids
• Apply diatomaceous earth around affected plants

**Natural Predators:**
• Release ladybugs or lacewings
• Plant fennel, dill, and yarrow to attract beneficial insects

**Prevention:**
• Avoid over-fertilizing with nitrogen
• Use reflective mulch to confuse aphids
• Regular inspection of plants"""

        elif any(word in prompt_lower for word in ["companion", "plant", "planting"]):
            return """🌿 **Companion Planting for Pest Control:**

**Pest-Repelling Plants:**
• Marigolds → repel nematodes, aphids
• Basil → repel flies, mosquitoes, thrips
• Catnip → stronger than DEET against mosquitoes
• Nasturtiums → trap crop for aphids, cucumber beetles

**Beneficial Attractors:**
• Fennel, dill → attract beneficial wasps
• Sunflowers → attract birds that eat pests
• Lavender → repels moths, fleas, mosquitoes"""

        elif any(word in prompt_lower for word in ["organic", "natural", "treatment"]):
            return """🌱 **Organic Pest Control Methods:**

**Physical Controls:**
• Row covers and barriers
• Sticky traps for flying insects
• Hand-picking larger pests

**Biological Controls:**
• Beneficial insects (ladybugs, lacewings)
• Bacillus thuringiensis (Bt) for caterpillars
• Predatory mites for spider mites

**Natural Sprays:**
• Neem oil for soft-bodied insects
• Insecticidal soap for aphids, mites
• Garlic and pepper sprays as deterrents"""

        else:
            return """🌱 **Organic Pest Management Tips:**

I'd be happy to help with organic pest control! I can provide advice on:
• Identifying common garden pests
• Natural and organic treatment methods
• Prevention strategies
• Companion planting
• Beneficial insects

Please ask me about specific pests or pest control topics, and I'll provide detailed organic solutions.

*Note: For full AI-powered responses, please ensure Ollama is running with the llama3.1:8b model.*"""
    
    def load_pest_database(self):
        """Load basic pest information for context"""
        return {
            "ants": {"family": "Formicidae", "treatment_priority": "medium"},
            "aphids": {"family": "Aphididae", "treatment_priority": "high"},
            "beetles": {"family": "Various", "treatment_priority": "medium"},
            "caterpillars": {"family": "Lepidoptera larvae", "treatment_priority": "high"},
            "earwigs": {"family": "Dermaptera", "treatment_priority": "low"},
            "grasshoppers": {"family": "Acrididae", "treatment_priority": "medium"},
            "moths": {"family": "Lepidoptera", "treatment_priority": "medium"},
            "slugs": {"family": "Gastropoda", "treatment_priority": "medium"},
            "snails": {"family": "Gastropoda", "treatment_priority": "medium"},
            "wasps": {"family": "Vespidae", "treatment_priority": "low"},
            "weevils": {"family": "Curculionidae", "treatment_priority": "high"}
        }
    
    def respond_to_question(self, message: str) -> str:
        """Main method to handle user questions with Ollama"""
        if not message.strip():
            return "Please ask me a question about organic pest management!"
        
        # Add message to conversation history
        self.conversation_history.append(f"User: {message}")
        
        # Check if Ollama is available
        if self.check_ollama_connection():
            response = self.call_ollama(message)
        else:
            print("⚠️ Ollama not available, using fallback responses")
            response = self.get_fallback_response(message)
        
        # Add response to conversation history
        self.conversation_history.append(f"Assistant: {response}")
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def get_pest_treatment(self, pest_name: str) -> str:
        """Get specific treatment advice for a pest using Ollama"""
        context = f"The user has identified a {pest_name} pest on their plants."
        prompt = f"What are the best organic treatment methods for {pest_name}? Please provide specific, actionable steps including timing, application methods, and safety considerations."
        
        if self.check_ollama_connection():
            return self.call_ollama(prompt, context)
        else:
            return self.get_fallback_response(f"treatment for {pest_name}")
    
    def get_pest_prevention(self, pest_name: str) -> str:
        """Get prevention advice for a specific pest"""
        context = f"The user wants to prevent {pest_name} infestations."
        prompt = f"How can I prevent {pest_name} from infesting my garden? Please provide organic prevention strategies including companion planting, cultural practices, and environmental modifications."
        
        if self.check_ollama_connection():
            return self.call_ollama(prompt, context)
        else:
            return self.get_fallback_response(f"prevention for {pest_name}")
    
    def get_followup_questions(self, pest_name: str) -> List[str]:
        """Generate relevant follow-up questions for a specific pest"""
        return [
            f"How can I prevent {pest_name} in the future?",
            f"What plants attract beneficial insects that eat {pest_name}?",
            f"When is the best time to treat {pest_name}?",
            f"Are there companion plants that repel {pest_name}?"
        ]
    
    def get_general_treatment_advice(self) -> str:
        """Get general organic pest control advice"""
        prompt = "What are the fundamental principles of organic pest management? Please provide a comprehensive overview of IPM strategies, beneficial insects, and natural treatment methods."
        
        if self.check_ollama_connection():
            return self.call_ollama(prompt)
        else:
            return self.get_fallback_response("general organic treatment")
    
    def get_general_prevention_advice(self) -> str:
        """Get general pest prevention advice"""
        prompt = "What are the best practices for preventing pest problems in an organic garden? Include soil health, plant diversity, and cultural practices."
        
        if self.check_ollama_connection():
            return self.call_ollama(prompt)
        else:
            return self.get_fallback_response("general prevention")
    
    def chat_with_context(self, message: str, pest_context: str = None, confidence: float = None) -> str:
        """Enhanced chat with pest identification context"""
        context = ""
        if pest_context:
            context = f"Recent pest identification: {pest_context}"
            if confidence:
                context += f" (confidence: {confidence:.1%})"
        
        if self.check_ollama_connection():
            return self.call_ollama(message, context)
        else:
            return self.get_fallback_response(message)