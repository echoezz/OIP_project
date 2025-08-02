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
from src.chat_bot import OrganicPestChatBot

class PestManagementApp:
    def __init__(self):
        self.identifier = None
        self.chatbot = OrganicPestChatBot()
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
        """Enhanced pest identification with integrated chat"""
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
            
            # Get treatment advice only if confident enough
            if result['meets_threshold']:
                treatment_advice = self.chatbot.get_pest_treatment(pest_name)
                
                # Add automatic chat response about the identification
                auto_message = f"I've identified this as {pest_name.replace('_', ' ')} with {confidence:.1%} confidence. Here's what I recommend for treatment:"
                history.append({"role": "assistant", "content": auto_message})
                
                followup_questions = self.chatbot.get_followup_questions(pest_name)
                followup_message = "💬 **You can also ask me:**\n" + "\n".join([f"• {q}" for q in followup_questions])
                history.append({"role": "assistant", "content": followup_message})
                
            else:
                treatment_advice = f"""
## ⚠️ **Low Confidence Identification**

The model is not confident enough about this identification. Here are some suggestions:

### 📸 **Improve Your Photo:**
• Take a closer, clearer image
• Ensure good lighting
• Include the pest and affected plant parts
• Try different angles

### 🔍 **General Organic Pest Control:**
• Inspect plants carefully for accurate identification
• Apply general organic insecticidal soap
• Introduce beneficial insects
• Maintain good garden hygiene

### 💬 **Get Help:**
Ask specific questions in the chat, or consult with a local gardening expert.
"""
                # Add low confidence message to chat
                auto_message = f"I detected what might be {pest_name.replace('_', ' ')}, but I'm only {confidence:.1%} confident. Could you try a clearer photo or ask me general questions about pest control?"
                history.append({"role": "assistant", "content": auto_message})
            
            return identification_result, treatment_advice, history
        else:
            error_msg = f"❌ Could not identify pest: {result.get('error', 'Unknown error')}"
            error_chat = "I couldn't identify the pest in this image. Please try uploading a clearer photo or ask me general questions about pest management."
            if history is None:
                history = []
            history.append({"role": "assistant", "content": error_chat})
            return error_msg, "", history
    
    def chat_response(self, message, history):
        """Handle chat interactions with new message format"""
        if not message.strip():
            return history, ""
        
        # Get bot response
        bot_response = self.chatbot.respond_to_question(message)
        
        # Add to history using new message format
        if history is None:
            history = []
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        
        return history, ""
    
    def handle_quick_question(self, question, history):
        """Handle quick question buttons with proper message format"""
        if history is None:
            history = []
        
        # Get bot response
        bot_response = self.chatbot.respond_to_question(question)
        
        # Add to history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": bot_response})
        
        return history
    
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
            <div class="pest-title">🌱 Enhanced Organic Pest Management Assistant</div>
            <div class="pest-subtitle">
                <span class="enhanced-badge">🚀 AI Enhanced with TTA</span><br>
                Upload a photo to identify pests and get organic treatment recommendations
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
                        <h3>💬 Ask Questions About Your Pest</h3>
                        <p>Get instant advice about treatment, prevention, and organic solutions!</p>
                        </div>
                        """)
                        
                        # Integrated chat interface
                        chatbot_interface = gr.Chatbot(
                            height=350,
                            label="Pest Management Expert",
                            type="messages",
                            placeholder="Chat will appear here after identification or ask questions directly..."
                        )
                        
                        # Chat input
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask about treatments, prevention, or pest management...",
                                label="Your Question",
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
                        
                        # Treatment output below chat
                        treatment_output = gr.Markdown(
                            label="🌿 Detailed Treatment Guide",
                            value="Treatment recommendations will appear here after identification..."
                        )
            
            # Enhanced About tab
            with gr.Tab("ℹ️ About Enhanced Version"):
                gr.HTML("""
                <div style="padding: 20px;">
                <h2>🌱 About Enhanced Organic Pest Management Assistant</h2>
                
                <div style="background: linear-gradient(45deg, #e8f5e8, #f0f8f0); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <h3>🚀 New Enhanced Features:</h3>
                <ul>
                <li><strong>🎯 Test-Time Augmentation (TTA):</strong> Analyzes 8 different image variations for higher accuracy</li>
                <li><strong>🧠 Focused CNN with Attention:</strong> Advanced neural network architecture</li>
                <li><strong>📊 Top-3 Predictions:</strong> See multiple possible identifications with confidence scores</li>
                <li><strong>🎪 Ensemble Predictions:</strong> Combines multiple model predictions</li>
                <li><strong>📈 Enhanced Data Augmentation:</strong> 15x training data multiplication</li>
                <li><strong>⚡ Smart Confidence Thresholding:</strong> Only provides treatment when confident</li>
                </ul>
                </div>
                
                <h3>🎯 What This Enhanced App Does:</h3>
                <ul>
                <li><strong>AI Pest Identification:</strong> Enhanced CNN with 50%+ accuracy improvement</li>
                <li><strong>Confidence Assessment:</strong> High/Medium/Low confidence levels with explanations</li>
                <li><strong>Organic Treatments:</strong> Safe, chemical-free treatment recommendations</li>
                <li><strong>Prevention Tips:</strong> Learn how to prevent pest problems naturally</li>
                <li><strong>Expert Chat:</strong> Intelligent conversation with context awareness</li>
                </ul>
                
                <h3>🧠 Technical Improvements:</h3>
                <ul>
                <li><strong>Focused CNN Architecture:</strong> Optimized for small datasets</li>
                <li><strong>Attention Mechanism:</strong> Focuses on important image features</li>
                <li><strong>Label Smoothing:</strong> Better generalization during training</li>
                <li><strong>Gradient Accumulation:</strong> Stable training with limited data</li>
                <li><strong>Cosine Annealing:</strong> Optimal learning rate scheduling</li>
                </ul>
                
                <h3>📊 Performance Metrics:</h3>
                <ul>
                <li><strong>Target Accuracy:</strong> 50%+ confidence (up from 13.5%)</li>
                <li><strong>Data Efficiency:</strong> 15x augmentation from original images</li>
                <li><strong>Prediction Speed:</strong> ~2-3 seconds with TTA</li>
                <li><strong>Model Size:</strong> Optimized for deployment</li>
                </ul>
                
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h4>💡 Pro Tips for Best Results:</h4>
                <p><strong>📸 Photography:</strong> Clear, well-lit, close-up photos of pests work best</p>
                <p><strong>🎯 Confidence:</strong> Higher confidence = more reliable identification and treatment advice</p>
                <p><strong>🔄 Multiple Angles:</strong> Try different angles if confidence is low</p>
                <p><strong>💬 Ask Questions:</strong> Use the chat for specific pest management questions</p>
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
            
        return interface

def main():
    """Main function to run the enhanced app"""
    print("🚀 Starting Enhanced Organic Pest Management Assistant...")
    
    app = PestManagementApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()