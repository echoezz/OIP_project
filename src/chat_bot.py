import json
import os
import re
from typing import Dict, List, Tuple
import random

class OrganicPestChatBot:
    def __init__(self):
        self.pest_database = self.load_pest_database()
        self.qa_responses = self.load_qa_responses()
        self.intent_patterns = self.load_intent_patterns()
        self.conversation_context = None  # Remember what user is talking about
    
    def load_pest_database(self):
        """Load pest database with error handling"""
        database_path = "data/pest_database.json"
        try:
            if os.path.exists(database_path):
                with open(database_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            return self.get_default_pest_database()
        except Exception as e:
            print(f"⚠️ Error loading pest database: {e}")
            return self.get_default_pest_database()
    
    def load_qa_responses(self):
        """Load QA responses with error handling"""
        qa_path = "data/qa_responses.json"
        try:
            if os.path.exists(qa_path):
                with open(qa_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            return self.get_default_qa_responses()
        except Exception as e:
            print(f"⚠️ Error loading QA responses: {e}")
            return self.get_default_qa_responses()
    
    def load_intent_patterns(self):
        """Load intent recognition patterns"""
        return {
            # Pest identification intents
            "pest_identification": [
                r"what.*pest.*this",
                r"identify.*pest",
                r"what.*bug.*this",
                r"what.*insect",
                r"help.*identify",
                r"what.*eating.*plant",
                r"pest.*identification"
            ],
            
            # Treatment/control intents
            "treatment": [
                r"how.*treat",
                r"how.*control",
                r"how.*get.*rid",
                r"how.*kill",
                r"treatment.*for",
                r"control.*pest",
                r"eliminate.*pest",
                r"organic.*treatment",
                r"natural.*remedy"
            ],
            
            # Prevention intents
            "prevention": [
                r"how.*prevent",
                r"stop.*pest",
                r"avoid.*pest",
                r"prevention",
                r"keep.*away",
                r"protect.*plant",
                r"pest.*proof"
            ],
            
            # Companion planting intents
            "companion_planting": [
                r"companion.*plant",
                r"plant.*together",
                r"what.*plant.*with",
                r"repel.*plant",
                r"beneficial.*plant",
                r"natural.*repellent.*plant"
            ],
            
            # Beneficial insects intents
            "beneficial_insects": [
                r"beneficial.*insect",
                r"good.*bug",
                r"natural.*predator",
                r"ladybug",
                r"lacewing",
                r"attract.*beneficial",
                r"biological.*control"
            ],
            
            # Timing intents
            "timing": [
                r"when.*apply",
                r"best.*time",
                r"timing",
                r"when.*treat",
                r"when.*spray",
                r"how.*often"
            ],
            
            # Organic methods intents
            "organic_methods": [
                r"organic.*method",
                r"natural.*way",
                r"chemical.*free",
                r"safe.*method",
                r"eco.*friendly",
                r"non.*toxic"
            ],
            
            # Specific pest intents
            "aphids": [
                r"aphid", r"plant.*lice", r"green.*bug", r"tiny.*green.*insect"
            ],
            "caterpillars": [
                r"caterpillar", r"worm", r"larvae", r"eating.*leaves"
            ],
            "spider_mites": [
                r"spider.*mite", r"red.*spider", r"tiny.*spider", r"webbing"
            ],
            "whiteflies": [
                r"whitefly", r"white.*fly", r"tiny.*white.*insect"
            ]
        }
    
    def analyze_intent(self, message: str) -> Tuple[str, str, float]:
        """Analyze user message to determine intent and extract pest mentions"""
        message_lower = message.lower()
        
        # Check for specific pest mentions
        detected_pest = None
        for pest_name, patterns in self.intent_patterns.items():
            if pest_name in self.pest_database:
                for pattern in patterns:
                    if re.search(pattern, message_lower):
                        detected_pest = pest_name
                        break
        
        # Check for general intents
        detected_intent = "general"
        max_confidence = 0.0
        
        intent_categories = ["treatment", "prevention", "companion_planting", 
                           "beneficial_insects", "timing", "organic_methods", 
                           "pest_identification"]
        
        for intent in intent_categories:
            if intent in self.intent_patterns:
                for pattern in self.intent_patterns[intent]:
                    if re.search(pattern, message_lower):
                        # Calculate confidence based on pattern specificity
                        confidence = len(pattern) / len(message_lower)
                        if confidence > max_confidence:
                            max_confidence = confidence
                            detected_intent = intent
        
        return detected_intent, detected_pest, max_confidence
    
    def respond_to_question(self, message: str) -> str:
        """Intelligent response to user questions"""
        if not message.strip():
            return "Please ask me about organic pest management!"
        
        # Analyze user intent
        intent, detected_pest, confidence = self.analyze_intent(message)
        
        # Update conversation context
        if detected_pest:
            self.conversation_context = detected_pest
        
        # Generate contextual response
        return self.generate_contextual_response(message, intent, detected_pest, confidence)
    
    def generate_contextual_response(self, message: str, intent: str, 
                                   detected_pest: str, confidence: float) -> str:
        """Generate intelligent contextual responses"""
        
        # Specific pest + treatment intent
        if detected_pest and intent == "treatment":
            return self.get_pest_treatment(detected_pest)
        
        # Specific pest + prevention intent  
        elif detected_pest and intent == "prevention":
            return self.get_pest_prevention(detected_pest)
        
        # General treatment questions
        elif intent == "treatment":
            return self.get_general_treatment_advice()
        
        # Prevention questions
        elif intent == "prevention":
            return self.get_general_prevention_advice()
        
        # Companion planting questions
        elif intent == "companion_planting":
            return self.get_companion_planting_advice()
        
        # Beneficial insects questions
        elif intent == "beneficial_insects":
            return self.get_beneficial_insects_advice()
        
        # Timing questions
        elif intent == "timing":
            return self.get_timing_advice()
        
        # Organic methods questions
        elif intent == "organic_methods":
            return self.get_organic_methods_advice()
        
        # Pest identification questions
        elif intent == "pest_identification":
            return self.get_identification_help()
        
        # Context-aware responses (user mentioned pest before)
        elif self.conversation_context:
            return self.get_contextual_followup(message, self.conversation_context)
        
        # Fallback to keyword matching
        else:
            return self.get_keyword_response(message)
    
    def get_pest_treatment(self, pest_name: str) -> str:
        """Get specific treatment for identified pest"""
        if pest_name in self.pest_database:
            pest_info = self.pest_database[pest_name]
            
            response = f"## 🌿 Organic Treatment for {pest_info['name']}\n\n"
            response += f"**Description:** {pest_info['description']}\n\n"
            
            if 'organic_treatments' in pest_info:
                response += "### 🧪 Recommended Treatments:\n"
                for i, treatment in enumerate(pest_info['organic_treatments'], 1):
                    response += f"{i}. {treatment}\n"
                response += "\n"
            
            response += "💡 **Pro Tip:** Apply treatments in early morning or evening for best results!"
            return response
        
        return f"I don't have specific treatment information for {pest_name}. Try asking about general organic treatments!"
    
    def get_pest_prevention(self, pest_name: str) -> str:
        """Get prevention advice for specific pest"""
        if pest_name in self.pest_database:
            pest_info = self.pest_database[pest_name]
            
            response = f"## 🛡️ Preventing {pest_info['name']}\n\n"
            
            if 'prevention' in pest_info:
                response += "### Prevention Strategies:\n"
                for i, prevention in enumerate(pest_info['prevention'], 1):
                    response += f"{i}. {prevention}\n"
                response += "\n"
            
            if 'natural_predators' in pest_info:
                response += "### 🐞 Encourage These Natural Predators:\n"
                response += ", ".join(pest_info['natural_predators'])
                response += "\n\n"
            
            response += "🌱 **Remember:** Prevention is always better than treatment!"
            return response
        
        return f"I don't have specific prevention tips for {pest_name}. Ask me about general prevention methods!"
    
    def get_general_treatment_advice(self) -> str:
        """General organic treatment advice"""
        return """## 🌿 General Organic Pest Treatments

### 🧪 Most Effective Organic Treatments:

1. **Insecticidal Soap** - Mix 2 tbsp mild liquid soap per quart of water
2. **Neem Oil** - Natural pesticide, apply in early morning or evening  
3. **Bacillus thuringiensis (Bt)** - Biological control for caterpillars
4. **Diatomaceous Earth** - Food-grade DE for crawling insects
5. **Essential Oil Sprays** - Peppermint, rosemary, or garlic-based
6. **Beneficial Insects** - Release ladybugs, lacewings, or predatory mites
7. **Physical Removal** - Hand-picking for larger pests

### ⏰ Application Tips:
• Apply in early morning (6-9 AM) or evening (6-8 PM)
• Avoid spraying during windy or rainy conditions
• Test on small area first
• Reapply every 7-10 days or after rain

**Ask me about specific pests for targeted advice!**"""
    
    def get_general_prevention_advice(self) -> str:
        """General prevention advice"""
        return """## 🛡️ Organic Pest Prevention Strategies

### 🌱 Build Natural Defenses:

1. **Companion Planting** - Use marigolds, nasturtiums, and herbs
2. **Beneficial Habitat** - Plant diverse flowers for natural predators  
3. **Healthy Soil** - Compost and organic matter boost plant immunity
4. **Crop Rotation** - Break pest lifecycles by changing plant families
5. **Regular Monitoring** - Weekly inspections catch problems early
6. **Proper Spacing** - Good air circulation prevents many issues
7. **Clean Garden** - Remove debris where pests overwinter

### 🏡 Create Pest-Resistant Garden:
• **Diversity is key** - Monocultures attract pests
• **Native plants** - Support local beneficial insects
• **Avoid over-fertilizing** - Excess nitrogen attracts pests
• **Encourage birds** - Install houses and water sources

**Prevention is 90% of successful organic gardening!**"""
    
    def get_companion_planting_advice(self) -> str:
        """Companion planting advice"""
        return """## 🌸 Companion Plants for Pest Control

### 🌻 Top Pest-Fighting Plants:

**🌼 Marigolds** - Repel aphids, whiteflies, nematodes
**🌿 Nasturtiums** - Trap crop for aphids, cucumber beetles  
**🌱 Basil** - Deters aphids, spider mites, thrips
**🍃 Mint** - Repels ants, mice, cabbage moths
**🧄 Garlic/Chives** - Natural fungicide and pest deterrent
**💜 Lavender** - Repels moths, fleas, mosquitoes
**🌿 Catnip** - More effective than DEET against mosquitoes

### 🎯 Strategic Planting:
• **Interplant** throughout vegetable garden
• **Border planting** around garden perimeter
• **Trap crops** - Plant nasturtiums to lure pests away
• **Herbs flowering** - Let dill, fennel, cilantro bloom for beneficials

### 🐝 Bonus Benefits:
• Attract pollinators and beneficial insects
• Provide herbs for cooking
• Add beauty and fragrance to garden

**Mix and match for maximum pest protection!**"""
    
    def get_beneficial_insects_advice(self) -> str:
        """Beneficial insects advice"""
        return """## 🐞 Attracting Beneficial Insects

### 🦋 Garden Heroes to Encourage:

**🐞 Ladybugs** - Eat aphids, scale insects, mites
**🦋 Lacewings** - Consume aphids, caterpillars, thrips
**🐝 Parasitic Wasps** - Attack many pest species  
**🕷️ Spiders** - General predators of flying pests
**🪲 Ground Beetles** - Night hunters of slugs, caterpillars

### 🌺 How to Attract Them:

1. **Plant Diverse Flowers** - Something blooming all season
2. **Native Plants** - Use indigenous species they recognize
3. **Herb Gardens** - Let herbs flower (dill, fennel, cilantro)
4. **Avoid Pesticides** - Even organic sprays can harm beneficials
5. **Provide Shelter** - Leave some garden areas undisturbed
6. **Water Sources** - Shallow dishes or plant saucers
7. **Purchase Releases** - Buy when pest pressure is high

### 🏠 Beneficial Insect Hotels:
• Bundle hollow stems (bamboo, reeds)
• Drill holes in wood blocks
• Stack stones or bricks
• Leave leaf litter in corners

**One beneficial insect can eat hundreds of pests!**"""
    
    def get_timing_advice(self) -> str:
        """Treatment timing advice"""
        return """## ⏰ Perfect Timing for Organic Treatments

### 🌅 Daily Timing:

**🌄 Early Morning (6-9 AM)** - BEST TIME
• Beneficial insects less active
• Plants can dry before heat
• Less wind interference
• Dew helps spray stick

**🌆 Evening (6-8 PM)** - Second best
• Avoid midday heat stress
• Less beneficial activity
• Treatments last overnight

**❌ Avoid Midday** - Can burn plants in hot sun

### 📅 Seasonal Timing:

**🌸 Spring** - Start prevention early
**☀️ Summer** - Weekly monitoring, early morning treatments  
**🍂 Fall** - Clean up, prepare for next year
**❄️ Winter** - Plan and order supplies

### 🌦️ Weather Considerations:
• **Before rain** - But not if heavy rain expected within 2 hours
• **Calm days** - Avoid windy conditions  
• **Dry leaves** - Don't spray wet foliage
• **Temperature** - Not below 50°F or above 85°F

### 📊 Treatment Frequency:
• **Prevention sprays** - Every 2 weeks
• **Active infestations** - Every 3-5 days
• **After rain** - Reapply organic treatments

**Timing can make or break your organic pest control!**"""
    
    def get_organic_methods_advice(self) -> str:
        """Organic methods overview"""
        return """## 🌱 Complete Organic Pest Management Methods

### 🎯 The Organic Approach:

**1. 🛡️ Prevention First**
• Build healthy soil and strong plants
• Use companion planting and diversity
• Encourage beneficial insects

**2. 🔍 Monitor Regularly** 
• Weekly garden inspections
• Early detection = easier control
• Know your pest lifecycles

**3. 🎪 Biological Controls**
• Beneficial insects and predators
• Microbial pesticides (Bt, beneficial nematodes)
• Trap crops and pheromone traps

**4. 🧪 Organic Sprays** (when needed)
• Insecticidal soap for soft-bodied pests
• Neem oil for various insects
• Essential oil sprays

**5. 🔧 Physical Methods**
• Hand-picking pests
• Row covers and barriers
• Sticky traps and copper strips

### ✅ Why Choose Organic:
• **Safe** for family, pets, beneficial insects
• **Sustainable** - doesn't harm soil or water
• **Effective** - builds long-term resistance
• **Cost-effective** - many ingredients at home

### 🚫 What to Avoid:
• Broad-spectrum pesticides
• Treatments that harm beneficials  
• Quick fixes that create dependency

**Organic gardening works WITH nature, not against it!**"""
    
    def get_identification_help(self) -> str:
        """Help with pest identification"""
        return """## 🔍 Pest Identification Help

### 📸 For Best Results:
• **Upload a clear photo** in the Pest Identification tab
• **Take close-up shots** of the pest and damage
• **Include affected plant parts** for context
• **Good lighting** helps AI accuracy

### 👀 What to Look For:

**🐛 Pest Characteristics:**
• Size and color
• Shape and body segments  
• Wings or no wings
• Where they're located on plant

**🍃 Damage Patterns:**
• Holes in leaves (chewing insects)
• Stippling or speckling (sucking insects)
• Webbing (spider mites)
• Sticky honeydew (aphids, scale)
• Yellowing or wilting

### 🔬 Common Garden Pests:

**🟢 Aphids** - Tiny, soft, clustered on new growth
**🐛 Caterpillars** - Worm-like, eat large holes in leaves
**🕷️ Spider Mites** - Tiny, cause stippling and webbing
**🦋 Whiteflies** - Small white flies, yellow when young

### 💡 Pro Tips:
• **Time of day matters** - Some pests hide during day
• **Check undersides** of leaves
• **Look for eggs** and larvae too
• **Note plant species** affected

**Upload a photo for instant AI identification!**"""
    
    def get_contextual_followup(self, message: str, pest_context: str) -> str:
        """Provide context-aware responses based on previous pest discussion"""
        message_lower = message.lower()
        
        # Treatment follow-ups
        if any(word in message_lower for word in ["how", "treat", "apply", "use"]):
            return self.get_pest_treatment(pest_context)
        
        # Prevention follow-ups
        elif any(word in message_lower for word in ["prevent", "avoid", "stop"]):
            return self.get_pest_prevention(pest_context)
        
        # Frequency questions
        elif any(word in message_lower for word in ["often", "when", "timing", "frequency"]):
            return f"For {pest_context}, I recommend checking your plants daily and applying treatments every 5-7 days until the problem is resolved. Early morning (6-9 AM) is the best time for most organic treatments."
        
        # Safety questions
        elif any(word in message_lower for word in ["safe", "pets", "children", "organic"]):
            return f"The organic treatments for {pest_context} I recommend are safe for family and pets when used as directed. Always wash vegetables before eating, and avoid spraying flowers that bees visit during blooming."
        
        # Default context response
        else:
            return f"I notice you're asking about {pest_context.replace('_', ' ')}. What specific aspect would you like to know more about? I can help with treatments, prevention, timing, or identification tips!"
    
    def get_keyword_response(self, message: str) -> str:
        """Fallback keyword-based responses"""
        message_lower = message.lower()
        
        keywords = {
            "hello": "Hello! I'm your organic pest management assistant. How can I help you today?",
            "help": "I can help you identify pests, recommend organic treatments, and provide prevention tips. What would you like to know?",
            "thanks": "You're welcome! Feel free to ask more questions about organic pest management anytime!",
            "organic": "Organic pest management uses natural methods that are safe for people, pets, and beneficial insects. What specific organic topic interests you?",
            "natural": "Natural pest control works with nature's own systems. I can suggest companion plants, beneficial insects, or organic sprays. What are you dealing with?"
        }
        
        for keyword, response in keywords.items():
            if keyword in message_lower:
                return response
        
        # Generic helpful response
        return """I'd love to help with your pest management question! Here are some things I can assist with:

🔍 **Pest Identification** - Upload photos for AI identification
🌿 **Organic Treatments** - Safe, effective natural solutions  
🛡️ **Prevention Tips** - Stop problems before they start
🌸 **Companion Planting** - Plants that repel pests
🐞 **Beneficial Insects** - Attract nature's pest controllers
⏰ **Treatment Timing** - When and how often to apply treatments

What specific pest problem can I help you solve?"""
    
    def get_followup_questions(self, pest_name: str) -> List[str]:
        """Generate relevant follow-up questions based on identified pest"""
        base_questions = [
            f"How do I prevent {pest_name.replace('_', ' ')} naturally?",
            f"When is the best time to treat {pest_name.replace('_', ' ')}?",
            f"What companion plants repel {pest_name.replace('_', ' ')}?",
            "How often should I apply organic treatments?",
            "Are these treatments safe for beneficial insects?"
        ]
        
        return base_questions
    
    def get_default_pest_database(self):
        """Default pest database if file missing"""
        return {
            "aphids": {
                "name": "Aphids",
                "description": "Small, soft-bodied insects that feed on plant sap",
                "organic_treatments": ["Insecticidal soap", "Neem oil", "Beneficial insects"],
                "prevention": ["Companion planting", "Beneficial habitat", "Avoid over-fertilizing"]
            }
        }
    
    def get_default_qa_responses(self):
        """Default QA responses if file missing"""
        return {
            "general": {
                "question": "General pest management",
                "answer": "I can help with organic pest identification and treatment advice."
            }
        }