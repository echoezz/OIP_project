import json
import re
from typing import Dict, List, Optional
import os

class OrganicPestChatBot:
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.pest_database = self.load_pest_database()
        self.qa_responses = self.load_qa_responses()
        
    def load_pest_database(self) -> Dict:
        """Load pest information database"""
        db_path = os.path.join(self.knowledge_base_path, "pest_database.json")
        try:
            with open(db_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_pest_database()
    
    def load_qa_responses(self) -> Dict:
        """Load Q&A responses"""
        qa_path = os.path.join(self.knowledge_base_path, "qa_responses.json")
        try:
            with open(qa_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_qa_responses()
    
    def create_default_pest_database(self) -> Dict:
        """Create default pest information database"""
        database = {
            "aphids": {
                "description": "Small, soft-bodied insects that feed on plant sap",
                "organic_treatments": [
                    "Spray with neem oil solution (2-3 tbsp per gallon of water)",
                    "Use insecticidal soap spray",
                    "Introduce beneficial insects like ladybugs",
                    "Plant companion plants like marigolds and catnip",
                    "Spray with garlic and pepper solution"
                ],
                "prevention": [
                    "Regular inspection of plants",
                    "Proper plant spacing for air circulation",
                    "Avoid over-fertilizing with nitrogen",
                    "Remove weeds that harbor aphids"
                ],
                "symptoms": "Curled leaves, sticky honeydew, yellowing plants"
            },
            "caterpillars": {
                "description": "Larvae of moths and butterflies that chew plant leaves",
                "organic_treatments": [
                    "Hand-pick caterpillars in early morning",
                    "Apply Bacillus thuringiensis (Bt) spray",
                    "Use row covers to prevent egg laying",
                    "Spray with spinosad (organic insecticide)",
                    "Encourage birds and beneficial wasps"
                ],
                "prevention": [
                    "Regular monitoring for eggs and larvae",
                    "Plant diverse crops to avoid large infestations",
                    "Use pheromone traps for moths",
                    "Maintain healthy soil for strong plants"
                ],
                "symptoms": "Chewed leaves, visible caterpillars, holes in foliage"
            },
            "spider_mites": {
                "description": "Tiny arachnids that cause stippling damage to leaves",
                "organic_treatments": [
                    "Spray with water to wash off mites",
                    "Apply neem oil or horticultural oil",
                    "Use predatory mites as biological control",
                    "Spray with rosemary or peppermint oil solution",
                    "Increase humidity around plants"
                ],
                "prevention": [
                    "Maintain adequate humidity",
                    "Avoid drought stress",
                    "Regular misting of plants",
                    "Remove dusty conditions"
                ],
                "symptoms": "Stippled leaves, fine webbing, yellowing foliage"
            },
            "whiteflies": {
                "description": "Small white flying insects that feed on plant undersides",
                "organic_treatments": [
                    "Use yellow sticky traps",
                    "Spray with neem oil or insecticidal soap",
                    "Release beneficial insects like Encarsia wasps",
                    "Apply diatomaceous earth around plants",
                    "Use reflective mulch to confuse whiteflies"
                ],
                "prevention": [
                    "Quarantine new plants",
                    "Regular inspection of leaf undersides",
                    "Avoid over-fertilizing",
                    "Remove infected plant debris"
                ],
                "symptoms": "White flying insects, yellowing leaves, sticky honeydew"
            }
        }
        
        # Save default database
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        with open(os.path.join(self.knowledge_base_path, "pest_database.json"), 'w') as f:
            json.dump(database, f, indent=2)
        
        return database
    
    def create_default_qa_responses(self) -> Dict:
        """Create default Q&A responses"""
        responses = {
            "greetings": [
                "Hello! I'm here to help you with organic pest management. What pest issue are you dealing with?",
                "Hi there! I can help you identify and treat pest problems organically. How can I assist you?",
                "Welcome! I specialize in organic pest control solutions. What can I help you with today?"
            ],
            "general_organic": [
                "Organic pest control focuses on natural, non-toxic methods that are safe for you, your family, and the environment.",
                "The key to organic pest management is prevention, biological control, and using natural deterrents.",
                "Healthy soil and plants are your first defense against pests. Strong plants can better resist pest damage."
            ],
            "prevention_tips": [
                "Regular plant inspection is crucial - catch problems early!",
                "Encourage beneficial insects by planting diverse flowers and herbs.",
                "Proper watering and spacing help prevent many pest issues.",
                "Crop rotation and companion planting are excellent prevention strategies."
            ],
            "when_to_treat": [
                "Start treatment as soon as you notice pest activity.",
                "Early morning or evening are the best times to apply organic treatments.",
                "Monitor plants regularly - prevention is easier than treatment."
            ]
        }
        
        # Save default responses
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        with open(os.path.join(self.knowledge_base_path, "qa_responses.json"), 'w') as f:
            json.dump(responses, f, indent=2)
        
        return responses
    
    def get_pest_treatment(self, pest_name: str) -> str:
        """Get detailed treatment information for identified pest"""
        pest_name = pest_name.lower().replace(" ", "_")
        
        if pest_name in self.pest_database:
            pest_info = self.pest_database[pest_name]
            
            response = f"🐛 **{pest_name.replace('_', ' ').title()} Treatment Guide**\n\n"
            response += f"**Description:** {pest_info['description']}\n\n"
            response += f"**Symptoms:** {pest_info['symptoms']}\n\n"
            
            response += "**🌱 Organic Treatment Options:**\n"
            for i, treatment in enumerate(pest_info['organic_treatments'], 1):
                response += f"{i}. {treatment}\n"
            
            response += "\n**🛡️ Prevention Tips:**\n"
            for i, prevention in enumerate(pest_info['prevention'], 1):
                response += f"{i}. {prevention}\n"
            
            response += "\n💡 **Tip:** Always test treatments on a small area first and apply during cooler parts of the day."
            
            return response
        else:
            return f"I don't have specific information about {pest_name} yet. However, here are some general organic pest control principles:\n\n" + \
                   self.get_general_advice()
    
    def get_general_advice(self) -> str:
        """Get general organic pest control advice"""
        return """🌿 **General Organic Pest Control Tips:**

1. **Identification First** - Properly identify the pest before treatment
2. **Monitor Regularly** - Check plants weekly for early detection
3. **Encourage Beneficials** - Plant flowers to attract helpful insects
4. **Healthy Soil** - Well-fed plants resist pests better
5. **Natural Barriers** - Use row covers and companion planting
6. **Biological Control** - Introduce beneficial predators
7. **Organic Sprays** - Neem oil, insecticidal soap, and horticultural oils
8. **Physical Removal** - Hand-picking for larger pests

Remember: Patience and persistence are key to successful organic pest management!"""
    
    def respond_to_question(self, question: str) -> str:
        """Generate response to user question"""
        question = question.lower().strip()
        
        # Greeting patterns
        if any(word in question for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            import random
            return random.choice(self.qa_responses['greetings'])
        
        # Organic/natural treatment questions
        if any(word in question for word in ['organic', 'natural', 'safe', 'non-toxic', 'chemical-free']):
            import random
            return random.choice(self.qa_responses['general_organic'])
        
        # Prevention questions
        if any(word in question for word in ['prevent', 'prevention', 'avoid', 'stop']):
            import random
            return random.choice(self.qa_responses['prevention_tips'])
        
        # When to treat questions
        if any(word in question for word in ['when', 'timing', 'best time']):
            import random
            return random.choice(self.qa_responses['when_to_treat'])
        
        # Treatment method questions
        if any(word in question for word in ['how', 'treat', 'get rid', 'control', 'kill']):
            return self.get_general_advice()
        
        # Specific pest questions
        for pest_name in self.pest_database.keys():
            if pest_name.replace('_', ' ') in question or pest_name in question:
                return self.get_pest_treatment(pest_name)
        
        # Default response
        return """I'm here to help with organic pest management! You can:

🔍 **Upload an image** for pest identification
❓ **Ask about specific pests** (aphids, caterpillars, spider mites, etc.)
🌱 **Get organic treatment advice**
🛡️ **Learn prevention strategies**

What would you like to know about organic pest control?"""
    
    def get_followup_questions(self, pest_name: str) -> List[str]:
        """Generate relevant follow-up questions"""
        return [
            f"How do I prevent {pest_name} in the future?",
            f"What companion plants help with {pest_name}?",
            f"When is the best time to treat {pest_name}?",
            f"Are there beneficial insects that control {pest_name}?",
            "What other organic pest control methods do you recommend?"
        ]