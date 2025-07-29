import gradio as gr
import torch
from PIL import Image
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.pest_identifier import PestIdentifier
from src.chat_bot import OrganicPestChatBot

class PestManagementApp:
    def __init__(self):
        self.identifier = None
        self.chatbot = OrganicPestChatBot()
        self.load_model()
    
    def load_model(self):
        """Load the trained pest identification model"""
        model_path = "models/saved_models/best_model.pth"
        classes_path = "models/saved_models/classes.json"
        
        if os.path.exists(model_path) and os.path.exists(classes_path):
            self.identifier = PestIdentifier(model_path, classes_path)
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ Model not found. Please train the model first.")
            self.identifier = None
    
    def identify_pest_and_get_treatment(self, image):
        """Main function that identifies pest and provides treatment"""
        if image is None:
            return "Please upload an image first.", "", ""
        
        if self.identifier is None:
            return "❌ Model not available. Please train the model first.", "", ""
        
        # Identify pest
        result = self.identifier.identify_pest(image)
        
        if result['success']:
            pest_name = result['pest_name']
            confidence = result['confidence']
            
            # Format identification result
            identification_result = f"""
🔍 **Pest Identified:** {pest_name.replace('_', ' ').title()}
📊 **Confidence:** {confidence:.1%}
            """
            
            # Get treatment advice
            treatment_advice = self.chatbot.get_pest_treatment(pest_name)
            
            # Get follow-up questions
            followup_questions = self.chatbot.get_followup_questions(pest_name)
            followup_text = "**💬 You can also ask:**\n" + "\n".join([f"• {q}" for q in followup_questions])
            
            return identification_result, treatment_advice, followup_text
        else:
            error_msg = f"❌ Could not identify pest: {result.get('error', 'Unknown error')}"
            return error_msg, "", ""
    
    def chat_response(self, message, history):
        """Handle chat interactions"""
        if not message.strip():
            return history, ""
        
        # Get bot response
        bot_response = self.chatbot.respond_to_question(message)
        
        # Add to history
        history.append([message, bot_response])
        
        return history, ""
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better mobile experience
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
        }
        .pest-subtitle {
            text-align: center;
            color: #5a8a50;
            font-size: 1.2em;
            margin-bottom: 1em;
        }
        .upload-box {
            border: 2px dashed #5a8a50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #f8f9f8;
        }
        .result-box {
            background-color: #f0f8f0;
            border-left: 4px solid #5a8a50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        """
        
        with gr.Blocks(css=css, title="🌱 Organic Pest Management Assistant") as interface:
            
            # Header
            gr.HTML("""
            <div class="pest-title">🌱 Organic Pest Management Assistant</div>
            <div class="pest-subtitle">Upload a photo to identify pests and get organic treatment recommendations</div>
            """)
            
            # Main tabs
            with gr.Tab("📸 Pest Identification"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Pest Image (JPG, PNG)",
                            sources=["upload", "camera"],
                            height=300
                        )
                        
                        identify_btn = gr.Button(
                            "🔍 Identify Pest & Get Treatment",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Example images section
                        gr.HTML("""
                        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9f8; border-radius: 8px;">
                        <h4>📋 Tips for Best Results:</h4>
                        <ul>
                        <li>Take clear, close-up photos of the pest</li>
                        <li>Ensure good lighting</li>
                        <li>Include the pest and affected plant parts</li>
                        <li>Multiple angles help with identification</li>
                        </ul>
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        identification_output = gr.Markdown(
                            label="🔍 Identification Result",
                            value="Upload an image to start pest identification..."
                        )
                        
                        treatment_output = gr.Markdown(
                            label="🌿 Organic Treatment Guide",
                            value="Treatment recommendations will appear here..."
                        )
                        
                        followup_output = gr.Markdown(
                            label="💬 Ask More Questions",
                            value="Follow-up questions will appear here..."
                        )
            
            # Chat tab
            with gr.Tab("💬 Ask Questions"):
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                <h3>🤔 Have Questions About Organic Pest Control?</h3>
                <p>Ask me anything about organic pest management, prevention, or treatment methods!</p>
                </div>
                """)
                
                chatbot_interface = gr.Chatbot(
                    height=400,
                    label="Organic Pest Management Expert"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about organic pest control, prevention tips, specific treatments...",
                        label="Your Question",
                        lines=2
                    )
                    
                with gr.Row():
                    send_btn = gr.Button("💬 Send", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat")
                
                # Quick question buttons
                with gr.Row():
                    gr.HTML("<h4>🚀 Quick Questions:</h4>")
                
                with gr.Row():
                    quick_btn1 = gr.Button("How to prevent aphids?", size="sm")
                    quick_btn2 = gr.Button("Best organic treatments?", size="sm")
                    quick_btn3 = gr.Button("When to apply treatments?", size="sm")
                    quick_btn4 = gr.Button("Companion planting tips?", size="sm")
            
            # Information tab
            with gr.Tab("ℹ️ About"):
                gr.HTML("""
                <div style="padding: 20px;">
                <h2>🌱 About Organic Pest Management Assistant</h2>
                
                <h3>🎯 What This App Does:</h3>
                <ul>
                <li><strong>Pest Identification:</strong> Upload photos to identify common garden pests</li>
                <li><strong>Organic Treatments:</strong> Get safe, chemical-free treatment recommendations</li>
                <li><strong>Prevention Tips:</strong> Learn how to prevent pest problems naturally</li>
                <li><strong>Expert Chat:</strong> Ask questions about organic pest management</li>
                </ul>
                
                <h3>🌿 Why Choose Organic?</h3>
                <ul>
                <li>Safe for family, pets, and beneficial insects</li>
                <li>Protects soil health and water quality</li>
                <li>Sustainable long-term solution</li>
                <li>Builds natural ecosystem balance</li>
                </ul>
                
                <h3>🔬 Supported Pests:</h3>
                <p>Currently identifies: Aphids, Caterpillars, Spider Mites, Whiteflies, and more...</p>
                
                <h3>📱 How to Use:</h3>
                <ol>
                <li>Take a clear photo of the pest or affected plant</li>
                <li>Upload the image in the "Pest Identification" tab</li>
                <li>Get instant identification and organic treatment advice</li>
                <li>Ask follow-up questions in the chat for more details</li>
                </ol>
                
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h4>💡 Pro Tip:</h4>
                <p>Regular monitoring and early intervention are key to successful organic pest management!</p>
                </div>
                </div>
                """)
            
            # Event handlers
            identify_btn.click(
                fn=self.identify_pest_and_get_treatment,
                inputs=[image_input],
                outputs=[identification_output, treatment_output, followup_output]
            )
            
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
            
            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot_interface, msg_input]
            )
            
            # Quick question handlers
            quick_btn1.click(
                fn=lambda: self.chat_response("How to prevent aphids?", []),
                outputs=[chatbot_interface, msg_input]
            )
            
            quick_btn2.click(
                fn=lambda: self.chat_response("What are the best organic treatments?", []),
                outputs=[chatbot_interface, msg_input]
            )
            
            quick_btn3.click(
                fn=lambda: self.chat_response("When to apply treatments?", []),
                outputs=[chatbot_interface, msg_input]
            )
            
            quick_btn4.click(
                fn=lambda: self.chat_response("Companion planting tips?", []),
                outputs=[chatbot_interface, msg_input]
            )
        
        return interface

def main():
    """Main function to run the app"""
    print("🌱 Starting Organic Pest Management Assistant...")
    
    app = PestManagementApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public link
        inbrowser=True,
        show_tips=True,
        enable_queue=True
    )

if __name__ == "__main__":
    main()