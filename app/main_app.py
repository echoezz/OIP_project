import gradio as gr
import torch
from PIL import Image
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))  # Add src directory to path

from src.pest_identifier import PestIdentifier
from src.rag_chat_bot import RAGChatBot

class PestManagementApp:
    def __init__(self):
        self.identifier = None
        self.chatbot = RAGChatBot(use_rag=True)
        self.load_model()
    
    def load_model(self):
        """Load the trained pest identification model"""
        model_path = "models/saved_models/73CNNDraft2.pth"
        classes_path = "models/saved_models/classes.json"
        
        if os.path.exists(model_path) and os.path.exists(classes_path):
            self.identifier = PestIdentifier(model_path, classes_path, enable_tta=True)
            print("✅ Enhanced model loaded successfully with TTA enabled!")
        else:
            print("⚠️ Model not found. Please train the model first.")
            self.identifier = None
    
    def identify_pest_and_get_treatment(self, image, history):
        """Enhanced pest identification without Ollama integration"""
        if image is None:
            return "Please upload an image first.", "", history
        
        if self.identifier is None:
            return "❌ Model not available. Please train the model first.", "", history
        
        # Get enhanced identification with top predictions
        result = self.identifier.get_top_predictions(image, top_k=3)
        
        if result['success']:
            pest_name = result['pest_name']
            confidence = result['confidence']
            confidence_level = result['confidence_level']
            method = result.get('method', 'standard')
            
            # Format enhanced identification result
            identification_result = f"""
## 🔍 **Enhanced Pest Identification Results**

### 🎯 **Primary Identification:**
**Pest:** {pest_name.replace('_', ' ').title()}  
**Confidence:** {confidence:.1%} ({confidence_level})  
**Method:** {method.replace('_', ' ').title()}  

### 📊 **Top 3 Predictions:**
"""
            
            for pred in result['top_predictions']:
                identification_result += f"{pred['rank']}. **{pred['pest_name'].replace('_', ' ').title()}** - {pred['confidence_percent']}\n"
            
            if method == 'TTA_enhanced':
                consistency = result.get('prediction_consistency', 0)
                num_aug = result.get('num_augmentations', 0)
                identification_result += f"\n**Prediction Consistency:** {consistency:.1%} (from {num_aug} augmentations)"
            
            identification_result += f"\n\n**Analysis:** {result['confidence_description']}"
            
            # Initialize chat history if needed
            if history is None:
                history = []
            
            # Provide basic treatment advice based on identification confidence
            if result['meets_threshold']:
                treatment_advice = f"""
## ✅ **Pest Identified Successfully**

The pest has been identified as **{pest_name.replace('_', ' ').title()}** with {confidence:.1%} confidence.

### 💬 **Get Treatment Recommendations:**
Use the chat interface below to ask specific questions about treatment methods for this pest.
"""
                
                # Add simple chat message about identification
                auto_message = f"✅ I've identified this as **{pest_name.replace('_', ' ').title()}** with {confidence:.1%} confidence. You can ask me specific questions about treatment in the chat below!"
                history.append({"role": "assistant", "content": auto_message})
                
            else:
                treatment_advice = f"""
## ⚠️ **Low Confidence Identification**

The model is not confident enough about this identification ({confidence:.1%} confidence).

### 📸 **Improve Your Photo:**
• Take a closer, clearer image
• Ensure good lighting
• Include the pest and affected plant parts
• Try different angles

### 🔍 **General Organic Pest Control:**
• Inspect plants carefully for accurate identification
• Apply general organic insecticidal soap as a safe first treatment
• Introduce beneficial insects to your garden
• Maintain good garden hygiene

### 💬 **Get Help:**
Use the chat interface to ask general questions about pest management, or consult with a local gardening expert for proper identification.
"""
                # Add low confidence message to chat
                auto_message = f"⚠️ I detected what might be **{pest_name.replace('_', ' ').title()}**, but I'm only {confidence:.1%} confident. Please try a clearer photo or ask me general questions about pest control in the chat!"
                history.append({"role": "assistant", "content": auto_message})
            
            return identification_result, treatment_advice, history
        else:
            error_msg = f"❌ Could not identify pest: {result.get('error', 'Unknown error')}"
            error_treatment = """
## ❌ **Identification Failed**

The model could not identify the pest in this image.

### 📸 **Try Again With:**
• A clearer, higher resolution image
• Better lighting conditions
• Close-up view of the pest
• Different angle or perspective

### 💬 **Alternative Help:**
Use the chat interface below to describe the pest or ask general questions about pest management.
"""
            error_chat = "❌ I couldn't identify the pest in this image. Please try uploading a clearer photo or ask me general questions about pest management in the chat."
            if history is None:
                history = []
            history.append({"role": "assistant", "content": error_chat})
            return error_msg, error_treatment, history
    
    def chat_response(self, message, history):
        """Handle chat interactions with new message format"""
        if not message.strip():
            return history, ""
        
        print(f"🗨️ User message: {message}")
        
        # Get bot response
        bot_response = self.chatbot.respond(message)
        
        print(f"🤖 Bot response received: {len(bot_response)} characters")
        print(f"🔍 Response preview: {bot_response[:100]}...")
        
        # Add to history using new message format
        if history is None:
            history = []
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        
        print(f"📝 Chat history updated, total messages: {len(history)}")
        
        return history, ""
    
    def handle_quick_question(self, question, history):
        """Handle quick question buttons with proper message format"""
        if history is None:
            history = []
        
        print(f"🚀 Quick question: {question}")
        
        # Get bot response
        bot_response = self.chatbot.respond(question)
        
        print(f"🤖 Quick response received: {len(bot_response)} characters")
        print(f"🔍 Response preview: {bot_response[:100]}...")
        
        # Add to history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": bot_response})
        
        print(f"📝 Quick chat history updated, total messages: {len(history)}")
        
        return history
    
    def test_chat_connection(self, history):
        """Test function to verify chat is working"""
        if history is None:
            history = []
        
        # Test connection
        is_connected = self.chatbot.check_ollama_connection()
        
        if is_connected:
            test_message = "Hello! Can you help me with organic pest control?"
            test_response = self.chatbot.respond(test_message)
            
            history.append({"role": "user", "content": "🧪 Test: " + test_message})
            history.append({"role": "assistant", "content": test_response})
            
            print(f"✅ Test successful - Response: {test_response[:50]}...")
        else:
            error_msg = "❌ Ollama connection test failed. Please ensure Ollama is running with 'ollama serve'"
            history.append({"role": "assistant", "content": error_msg})
            print(error_msg)
        
        return history
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = self.chatbot.get_system_status()
        
        status_text = "## 🔧 System Status\n\n"
        
        # Ollama status
        if status['ollama_connected']:
            status_text += "✅ **Ollama:** Connected and ready\n"
        else:
            status_text += "❌ **Ollama:** Not connected (using fallback responses)\n"
        
        # RAG status
        if status['rag_enabled']:
            if status['rag_functional']:
                status_text += "✅ **RAG System:** Enabled and functional\n"
                if 'rag_collection_info' in status and status['rag_collection_info']:
                    info = status['rag_collection_info']
                    status_text += f"   📊 Vector Database: {info.get('points_count', 0)} documents indexed\n"
            else:
                status_text += "⚠️ **RAG System:** Enabled but not functional\n"
        else:
            status_text += "❌ **RAG System:** Disabled (using basic responses)\n"
        
        # Model status
        if self.identifier:
            status_text += "✅ **Pest Identification Model:** Loaded and ready\n"
        else:
            status_text += "❌ **Pest Identification Model:** Not available\n"
        
        # Chat history
        status_text += f"📝 **Chat History:** {status['conversation_history_length']} messages\n"
        
        return status_text
    
    def initialize_rag_system(self):
        """Initialize or re-initialize the RAG system"""
        if not self.chatbot.use_rag:
            return "❌ RAG system is not enabled. Please restart the application with RAG support."
        
        try:
            success, message = self.chatbot.initialize_knowledge_base()
            if success:
                return f"✅ **RAG System Initialized Successfully**\n\n{message}\n\nThe system can now provide enhanced responses based on your pest management documents."
            else:
                return f"❌ **RAG Initialization Failed**\n\n{message}"
        except Exception as e:
            return f"❌ **RAG Initialization Error**\n\nError: {str(e)}"

    def reset_vector_database(self):
        """Reset the vector database to clear old indexed data"""
        if not self.chatbot.use_rag:
            return "❌ RAG system is not enabled. Please restart the application with RAG support."
        
        try:
            success, message = self.chatbot.reset_vector_database()
            if success:
                return f"✅ **Vector Database Reset Successfully**\n\n{message}\n\nYou can now re-initialize the RAG system with fresh data."
            else:
                return f"❌ **Vector Database Reset Failed**\n\n{message}"
        except Exception as e:
            return f"❌ **Vector Database Reset Error**\n\nError: {str(e)}"
    
    def create_interface(self):
        """Create the enhanced Gradio interface"""
        
        # Enhanced CSS
        css = """
        .gradio-container {
            max-width: 100% !important;
            margin: 0 auto !important;
        }
        .pest-title {
            text-align: center;
            color: #2d5a27;
            font-size: 2.5em;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .pest-subtitle {
            text-align: center;
            color: #5a8a50;
            font-size: 1.2em;
            margin-bottom: 1em;
        }
        .enhanced-badge {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .confidence-high { color: #4CAF50; font-weight: bold; }
        .confidence-medium { color: #FF9800; font-weight: bold; }
        .confidence-low { color: #F44336; font-weight: bold; }
        """
        
        with gr.Blocks(css=css, title="🌱 Enhanced Organic Pest Management Assistant") as interface:
            
            # Header with enhancement badge
            gr.HTML("""
            <div class="pest-title">🌱 AI-Powered Organic Pest Management Assistant</div>
            <div class="pest-subtitle">
                <span class="enhanced-badge">🤖 Powered by Ollama LLaMA 3.2:3B</span><br>
                Upload a photo to identify pests and chat with AI for organic treatment recommendations
            </div>
            """)
            
            # Main tabs
            with gr.Tab("📸 Enhanced Pest Identification"):
                with gr.Row():
                    # Left column - Image upload and identification
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Pest Image (JPG, PNG)",
                            sources=["upload", "webcam"],
                            height=300
                        )
                        
                        identify_btn = gr.Button(
                            "🔍 Enhanced Identify & Get Treatment",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Identification results
                        identification_output = gr.Markdown(
                            label="🔍 Enhanced Identification Result",
                            value="Upload an image to start enhanced pest identification..."
                        )
                        
                        # Enhanced tips section
                        gr.HTML("""
                        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9f8; border-radius: 8px;">
                        <h4>📋 Enhanced AI Tips:</h4>
                        <ul>
                        <li><strong>🎯 Test-Time Augmentation:</strong> AI analyzes 8 different views of your image</li>
                        <li><strong>📸 Photo Quality:</strong> Clear, close-up photos work best</li>
                        <li><strong>💡 Lighting:</strong> Natural light provides best results</li>
                        <li><strong>🔍 Focus:</strong> Include both pest and affected plant parts</li>
                        <li><strong>📐 Angles:</strong> Multiple angles help with uncertain cases</li>
                        </ul>
                        <p><strong>🚀 New:</strong> Enhanced CNN with attention mechanism for better accuracy!</p>
                        </div>
                        """)
                    
                    # Right column - Chat interface and treatment info
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="text-align: center; margin-bottom: 15px;">
                        <h3>💬 AI Chat Assistant</h3>
                        <p>Ask the AI anything about organic pest management! Powered by Ollama LLaMA 3.2:3B</p>
                        </div>
                        """)
                        
                        # Integrated chat interface
                        chatbot_interface = gr.Chatbot(
                            height=350,
                            label="🤖 Ollama AI Pest Expert",
                            type="messages",
                            placeholder="Chat with the AI about pest management - powered by Ollama LLaMA 3.2:3B..."
                        )
                        
                        # Chat input
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask the AI anything about organic pest management...",
                                label="Chat with AI",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("💬 Send", variant="secondary", scale=1)
                        
                        # Quick question buttons
                        with gr.Row():
                            quick_btn1 = gr.Button("🌿 Organic Treatments", size="sm")
                            quick_btn2 = gr.Button("🛡️ Prevention Tips", size="sm")
                        
                        with gr.Row():
                            quick_btn3 = gr.Button("⏰ When to Apply", size="sm")
                            quick_btn4 = gr.Button("🌸 Companion Plants", size="sm")
                        
                        with gr.Row():
                            test_btn = gr.Button("🧪 Test Ollama Connection", size="sm", variant="secondary")
                        
                        # Treatment output below chat
                        treatment_output = gr.Markdown(
                            label="🌿 Detailed Treatment Guide",
                            value="Treatment recommendations will appear here after identification..."
                        )
            
            # RAG System Management Tab
            with gr.Tab("🧠 RAG Knowledge System"):
                gr.HTML("""
                <div style="padding: 20px;">
                <h2>🧠 RAG (Retrieval-Augmented Generation) System</h2>
                <p>The RAG system enhances AI responses by searching through your pest management documents and research papers.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <h3>📊 System Status</h3>
                        <p>Check the current status of all system components:</p>
                        """)
                        
                        status_btn = gr.Button("🔍 Check System Status", variant="secondary")
                        status_output = gr.Markdown(
                            value="Click 'Check System Status' to see current system information...",
                            label="System Status"
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <h3>⚙️ RAG Initialization</h3>
                        <p>Initialize or refresh the knowledge base from your documents:</p>
                        """)
                        
                        init_btn = gr.Button("🚀 Initialize RAG System", variant="primary")
                        reset_btn = gr.Button("🗑️ Reset Vector Database", variant="secondary")
                        
                        init_output = gr.Markdown(
                            value="Click 'Initialize RAG System' to process your knowledge base documents...",
                            label="RAG Initialization Status"
                        )
                
                gr.HTML("""
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h4>📚 How RAG Works:</h4>
                <ul>
                <li><strong>Document Processing:</strong> Only the Appendix-4-Pest-Management-Framework.pdf is processed</li>
                <li><strong>Vector Embeddings:</strong> Document text is converted to vector representations using AI</li>
                <li><strong>Semantic Search:</strong> When you ask questions, relevant document sections are found</li>
                <li><strong>Enhanced Responses:</strong> AI combines found information with its knowledge for better answers</li>
                <li><strong>Summarization:</strong> Long chunks are summarized to focus on query-relevant content</li>
                </ul>
                </div>
                
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <h4>🔧 Requirements:</h4>
                <ul>
                <li><strong>Qdrant Vector Database:</strong> Must be running on localhost:6333</li>
                <li><strong>Knowledge Base:</strong> Appendix-4-Pest-Management-Framework.pdf in knowledge_base/ folder</li>
                <li><strong>Internet Connection:</strong> Required for downloading embedding models (first time only)</li>
                </ul>
                
                <h4>📝 Setup Instructions:</h4>
                <p>1. Install and run Qdrant: <code>docker run -p 6333:6333 qdrant/qdrant</code></p>
                <p>2. Ensure Appendix-4-Pest-Management-Framework.pdf is in the knowledge_base/ folder</p>
                <p>3. Click 'Reset Vector Database' to clear old data (if needed)</p>
                <p>4. Click 'Initialize RAG System' to process the PDF document</p>
                <p>5. Use the chat interface - responses will now include summarized information from the PDF!</p>
                </div>
                """)
            
            # Enhanced About tab
            with gr.Tab("ℹ️ About AI-Powered Version"):
                gr.HTML("""
                <div style="padding: 20px;">
                <h2>🌱 About AI-Powered Organic Pest Management Assistant</h2>
                
                <div style="background: linear-gradient(45deg, #e8f5e8, #f0f8f0); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <h3>🤖 AI-Powered Features:</h3>
                <ul>
                <li><strong>🧠 Ollama LLaMA 3.1 8B:</strong> Advanced language model for intelligent conversations</li>
                <li><strong>💬 Dynamic Chat:</strong> Real-time, context-aware responses about pest management</li>
                <li><strong>� Expert Knowledge:</strong> Trained on organic pest control best practices</li>
                <li><strong>🔄 Conversation Memory:</strong> Remembers context throughout your session</li>
                <li><strong>🌿 Organic Focus:</strong> Specialized in chemical-free, sustainable solutions</li>
                <li><strong>⚡ Fallback System:</strong> Works even when Ollama is offline</li>
                </ul>
                </div>
                
                <h3>🎯 What This AI Assistant Does:</h3>
                <ul>
                <li><strong>🤖 Intelligent Chat:</strong> Natural language conversations about pest management</li>
                <li><strong>🔍 Pest Identification:</strong> Visual identification with AI-powered analysis</li>
                <li><strong>🌱 Organic Treatments:</strong> Personalized, chemical-free treatment recommendations</li>
                <li><strong>🛡️ Prevention Strategies:</strong> Proactive advice to prevent pest problems</li>
                <li><strong>🌸 Companion Planting:</strong> Expert advice on beneficial plant combinations</li>
                <li><strong>📅 Timing Guidance:</strong> When and how to apply treatments effectively</li>
                </ul>
                
                <h3>🚀 AI Technology Stack:</h3>
                <ul>
                <li><strong>Language Model:</strong> Ollama LLaMA 3.1 8B (locally hosted)</li>
                <li><strong>Computer Vision:</strong> MobileNet for pest identification</li>
                <li><strong>Framework:</strong> TensorFlow + Gradio interface</li>
                <li><strong>Knowledge Base:</strong> Organic pest management expertise</li>
                <li><strong>Context Awareness:</strong> Maintains conversation history</li>
                </ul>
                
                <h3>� Setup Requirements:</h3>
                <ul>
                <li><strong>Ollama Installation:</strong> Install Ollama and download llama3.1:8b model</li>
                <li><strong>Model Running:</strong> Ensure Ollama is running on http://localhost:11434</li>
                <li><strong>Fallback Mode:</strong> App works with pre-defined responses if Ollama is unavailable</li>
                </ul>
                
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h4>💡 Pro Tips for Best AI Experience:</h4>
                <p><strong>🤖 Natural Conversation:</strong> Ask questions in natural language - the AI understands context</p>
                <p><strong>📸 Image + Chat:</strong> Upload a pest photo, then ask specific follow-up questions</p>
                <p><strong>🔄 Follow-up Questions:</strong> Ask for clarification, alternative treatments, or timing advice</p>
                <p><strong>🌱 Specify Your Garden:</strong> Mention your location, plants, or garden type for better advice</p>
                <p><strong>� Learn More:</strong> Ask "why" questions to understand the science behind recommendations</p>
                </div>
                
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <h4>⚙️ Ollama Setup Instructions:</h4>
                <p>1. Install Ollama from <strong>https://ollama.ai</strong></p>
                <p>2. Run: <code>ollama pull llama3.1:8b</code></p>
                <p>3. Start Ollama service: <code>ollama serve</code></p>
                <p>4. The app will automatically detect and use Ollama for enhanced AI responses!</p>
                </div>
                </div>
                """)
            
            # Event handlers for integrated pest identification tab
            identify_btn.click(
                fn=self.identify_pest_and_get_treatment,
                inputs=[image_input, chatbot_interface],
                outputs=[identification_output, treatment_output, chatbot_interface]
            )
            
            # Chat functionality in pest identification tab
            send_btn.click(
                fn=self.chat_response,
                inputs=[msg_input, chatbot_interface],
                outputs=[chatbot_interface, msg_input]
            )
            
            msg_input.submit(
                fn=self.chat_response,
                inputs=[msg_input, chatbot_interface],
                outputs=[chatbot_interface, msg_input]
            )
            
            # Quick question handlers for pest identification tab
            quick_btn1.click(
                fn=lambda hist: self.handle_quick_question("What are the best organic treatments?", hist),
                inputs=[chatbot_interface],
                outputs=[chatbot_interface]
            )
            
            quick_btn2.click(
                fn=lambda hist: self.handle_quick_question("How can I prevent pest problems naturally?", hist),
                inputs=[chatbot_interface],
                outputs=[chatbot_interface]
            )
            
            quick_btn3.click(
                fn=lambda hist: self.handle_quick_question("When is the best time to apply treatments?", hist),
                inputs=[chatbot_interface],
                outputs=[chatbot_interface]
            )
            
            quick_btn4.click(
                fn=lambda hist: self.handle_quick_question("What companion plants help repel pests?", hist),
                inputs=[chatbot_interface],
                outputs=[chatbot_interface]
            )
            
            test_btn.click(
                fn=self.test_chat_connection,
                inputs=[chatbot_interface],
                outputs=[chatbot_interface]
            )
            
            # RAG System Management event handlers
            status_btn.click(
                fn=self.get_system_status,
                inputs=[],
                outputs=[status_output]
            )
            
            init_btn.click(
                fn=self.initialize_rag_system,
                inputs=[],
                outputs=[init_output]
            )
            
            reset_btn.click(
                fn=self.reset_vector_database,
                inputs=[],
                outputs=[init_output]
            )
            
        return interface

def main():
    """Main function to run the AI-powered app"""
    print("🤖 Starting AI-Powered Organic Pest Management Assistant...")
    print("🔧 Connecting to Ollama LLaMA 3.2:3B for intelligent chat...")
    
    app = PestManagementApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()