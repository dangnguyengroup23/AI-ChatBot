from flask import Flask, request, jsonify, render_template
from dp import create_table, insert_conversation
import torch
from model import ANN
from nltk_utils import bagOfWord, tokenize
import random
import json
import logging
import re
from datetime import datetime
from typing import Dict,List,Optional,Tuple
from flask_cors import CORS
import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)

CORS(app)

class Chatbot:
    def __init__(self,interns_file: str ='intents.json',model_file:str ='data.pth'):
        self.intents = None
        self.model = None
        self.all_words = None
        self.tag = None
        self.confident_threshold = 0.75
        self.history_conversation = {}

        self._load_intents(interns_file)
        self._load_model(model_file)

    def _load_intents(self,intents_file:str) ->None:
        try:
            with open(intents_file,'r',encoding='utf-8') as f:
                self.intents = json.load(f)
            logger.info(f"Successfully loaded intents from {intents_file}")
        except FileNotFoundError:
            logger.error(f"Intents file {intents_file} not found")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in intents file: {e}")
            raise
    
    def _load_model(self,model_file:str)->None:
        try:
            data = torch.load(model_file,map_location=torch.device('cpu'))

            input_size = data['input_size']
            hidden_size = data['hidden_size']
            output_size = data['output_size']
            self.all_words = data['all_words']
            self.tags = data['tags']
            model_state = data['model_state']

            self.model = ANN(input_size,hidden_size,output_size)
            self.model.load_state_dict(model_state)
            self.model.eval()
            logger.info(f"Successfully loaded model from {model_file}")
            logger.info(f"Model architecture: {self.model.getArchitectureInformation()}")
        except FileNotFoundError:
            logger.error(f"Model file {model_file} not found")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _preprocess_message(self,message:str) -> str:
        if not message or not isinstance(message,str):
            return ""
        message = re.sub(r'\s+', ' ',message.strip())

        message = re.sub(r'[<>{}]', '',message)

        return message
    
    def _get_prediction(self,message:str)-> Tuple[str,float]:
        try:
            tokens = tokenize(message)
            X = bagOfWord(tokens,self.all_words)
            X = X.reshape(1,X.shape[0])
            X = torch.from_numpy(X).float()

            with torch.no_grad():
                output = self.model(X)
                _,predict = torch.max(output,dim=1)
                tag = self.tags[predict.item()]

                probs = torch.softmax(output,dim=1)
                confidence = probs[0][predict.item()].item()
            return tag,confidence
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 'error',0.0
    
    def _get_response_for_tag(self,tag:str,sentiment:str =None)->str:
        for intent in self.intents['intents']:
            if tag == intent['tag']:
                if isinstance(intent['responses'],dict) and sentiment in intent['responses']:
                    return random.choice(intent['responses'][sentiment])
                else :
                    return random.choice(intent['responses'])
        
        return "I'm not sure how to respond to that."
    
    def _handle_low_confident(self,message:str)->str:
        if len(message.split())<=2:
            return "Hmm, could you tell me more? That was a bit short to understand."
        if "error_handling" in self.intents:
            unclear_response = self.intents['error_handling'].get('unclear_input',[])
            if unclear_response:
                return random.choice(unclear_response)
        
        return "I'm not sure I understand. Could you rephrase that or ask something else?"
    
    def _update_conversation_history(self,session_id: str,meesage : str,response: str) ->None:
        from dp import insert_conversation
        timestamp = datetime.now().isoformat()
        if session_id not in self.history_conversation:
            self.history_conversation[session_id]=[]
        
        self.history_conversation[session_id].append({
            'timestamp' : timestamp,
            'user_message' : meesage,
            'bot_response' : response
        })
        if len(self.history_conversation[session_id])>10:
            self.history_conversation[session_id] = self.history_conversation[session_id][-10:]

        insert_conversation(session_id,timestamp,meesage,response)

    def _detect_sentiment(self, message: str) -> str:
        positive_words = ['happy', 'good', 'great', 'awesome', 'fantastic', 'excellent', 'amazing', 'joy', 'love', 'nice']
        negative_words = ['sad', 'bad', 'terrible', 'awful', 'horrible', 'angry', 'frustrated', 'hate', 'depressed', 'upset']
        negation_words = ['not', "don't", "didn't", "isn't", "wasn't", "can't", "couldn't", "won't"]

        message_lower = message.lower()
        tokens = message_lower.split()

        found_positive = False
        found_negative = False

        for i, word in enumerate(tokens):
            window = tokens[max(0, i - 2):i]  # check up to 2 words behind
            is_negated = any(neg in window for neg in negation_words)

            if word in positive_words:
                if is_negated:
                    found_negative = True
                else:
                    found_positive = True
            elif word in negative_words:
                if is_negated:
                    found_positive = True
                else:
                    found_negative = True

        if found_positive and not found_negative:
            return 'positive'
        elif found_negative and not found_positive:
            return 'negative'
        elif found_positive and found_negative:
            return 'neutral'
        else:
            return 'neutral'


    def get_response(self,message:str,session_id: str = None) ->Dict:

        try:
            clean_message = self._preprocess_message(message)

            if not clean_message:
                return {
                    "reply": "I didn't receive any message. Please try again!",
                    "confidence": 0.0,
                    "tag": "error"
                }

            predict_tag,confidence = self._get_prediction(clean_message)

            if confidence>self.confident_threshold:
        

                sentiment = self._detect_sentiment(clean_message)

                if predict_tag=='mood':
                    response = self._get_response_for_tag(predict_tag,sentiment)
                else:
                    response = self._get_response_for_tag(predict_tag)
            else:
                response = self._handle_low_confident(clean_message)
                predict_tag = 'unclear'
            
            if session_id:
                self._update_conversation_history(session_id,clean_message,response)

            return {
                "reply": response,
                "confidence": confidence,
                "tag": predict_tag
            }
        except Exception as e:
            logger.error(f"Error in get_response: {e}")

            return {
                "reply": "I'm experiencing some technical difficulties. Please try again in a moment.",
                "confidence": 0.0,
                "tag": "error"
            }

try:
    chatbot = Chatbot()
    logger.info("Chatbot engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    chatbot = None

@app.route("/")
def home():
    return render_template('index.html')
@app.route('/chat',methods =['POST'])
def chat():
    try:
        if not chatbot:
            return jsonify({
                "error": "Chatbot is not properly initialized"
            }) ,500
        
        if not request.json:
            return jsonify({
                "error": "No JSON data provided"
            }),400
        
        message = request.json.get("message", "").strip()
        session_id = request.json.get("session_id", None)

        if not message:
            return jsonify({
                "error": "No message provided"
            }), 400
        

        if len(message) > 1000:
            return jsonify({
                "error": "Message too long. Please keep it under 1000 characters."
            }), 400
        
        response_data = chatbot.get_response(message,session_id)

        response_data['timestamp'] = datetime.now().isoformat()

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "reply": "I'm sorry, I encountered an error. Please try again."
        }), 500
@app.route("/health", methods=["GET"])
def health_check():
    
    try:
        if chatbot and chatbot.model:
            return jsonify({
                "status": "healthy",
                "model_loaded": True,
                "intents_loaded": len(chatbot.intents.get("intents", [])) if chatbot.intents else 0
            })
        else:
            return jsonify({
                "status": "unhealthy",
                "model_loaded": False
            }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/stats", methods=["GET"])
def get_stats():
   
    try:
        if not chatbot:
            return jsonify({"error": "Chatbot not initialized"}), 500
        
        total_conversations = len(chatbot.conversation_history)
        total_intents = len(chatbot.intents.get("intents", [])) if chatbot.intents else 0
        
        return jsonify({
            "total_conversations": total_conversations,
            "total_intents": total_intents,
            "confidence_threshold": chatbot.confidence_threshold
        })
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/model-info", methods=["GET"])
def get_model_info():
 
    try:
        if not chatbot or not chatbot.model:
            return jsonify({"error": "Model not loaded"}), 500
        
        model_info = chatbot.model.get_architecture_info()
        model_info.update({
            "confidence_threshold": chatbot.confidence_threshold,
            "vocabulary_size": len(chatbot.all_words) if chatbot.all_words else 0,
            "num_intents": len(chatbot.tags) if chatbot.tags else 0,
            "intent_tags": chatbot.tags if chatbot.tags else []
        })
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error in model-info endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
 
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":

    app.run(
        debug=True,
        host='127.0.0.1',  
        port=5000,
        threaded=True  
    )


if __name__ == "__main__":
    create_table()
    app.run(debug=True)
