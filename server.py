from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pyngrok import ngrok
from pymongo import MongoClient
from bson import ObjectId
from openai import OpenAI
import os
import datetime
import subprocess
import json
from dotenv import load_dotenv
load_dotenv()



# Initialize Flask app
app = Flask(__name__)
CORS(app)
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize OpenAI client
client = OpenAI()

# MongoDB setup
mongo_uri = os.getenv('MONGO_URI')
db_client = MongoClient(mongo_uri)
db = db_client['transcription_db']
collection = db['transcriptions']

def ensure_wav_format(file_path):
    """Convert audio file to wav format if needed"""
    wav_path = os.path.splitext(file_path)[0] + '.wav'
    subprocess.call(['ffmpeg', '-i', file_path, wav_path, '-y'])
    return wav_path

def generate_meeting_notes(transcription):
    """Generate comprehensive meeting notes using OpenAI's Chat API"""
    
    system_prompt = (
        "You are an advanced AI assistant specializing in analyzing meeting transcriptions. "
        "You provide structured, comprehensive meeting notes that capture the most important aspects of discussions. "
        "Ensure the notes are organized into key sections to facilitate understanding and action."
    )
    
    user_prompt = (
        "Please analyze the following meeting transcription and generate the following sections:\n\n"
        "1. **Summary of Meeting:** A concise overview of the main points discussed.\n"
        "2. **Key Decisions:** Specific decisions made during the meeting.\n"
        "3. **Open Questions:** Any unresolved or follow-up questions raised.\n"
        "4. **Action Items:** A list of explicit and implicit tasks assigned, with responsible parties.\n"
        "5. **Next Steps:** Outline what needs to be done after the meeting.\n\n"
        "Format your response in clear, labeled sections.\n\n"
        "Meeting Transcription:\n\n"
        f"{transcription}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content

@app.route('/transcribe', methods=['POST'])
def transcribe_conversation():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        # Save uploaded file
        audio_file = request.files['audio']
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join('uploads', filename)
        audio_file.save(filepath)

        # Transcribe using OpenAI's Whisper API
        with open(filepath, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        print("Transcription:", transcription)
        print("Transcription Text", transcription.text)

        # Generate meeting notes
        meeting_notes = generate_meeting_notes(transcription.text)
        print("Meeting Notes:", meeting_notes)

        # Save transcription
        # Generate timestamp format for transcription
        formatted_transcription = f"[00:00 - END] Transcription: {transcription.text}\n"

        # Generate summary and action items
        summary_and_actions = meeting_notes

        # Save results
        result_filename = f'result_{filename}.json'
        result_path = os.path.join('results', result_filename)
        
        result_data = {
            "transcription": formatted_transcription,
            "summary_and_actions": summary_and_actions
        }

        with open(result_path, 'w') as f:
            json.dump(result_data, f)

        # Store in MongoDB
        document = {
            "filename": filename,
            "transcription": formatted_transcription,
            "summary_and_actions": summary_and_actions,
            "timestamp": datetime.datetime.utcnow()
        }
        inserted_id = collection.insert_one(document).inserted_id

        # Clean up
        os.remove(filepath)

        return jsonify({
            "message": "Transcription and summary generated successfully",
            "transcription": formatted_transcription,
            "summary_and_actions": summary_and_actions
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_transcriptions', methods=['GET'])
def get_transcriptions():
    try:
        transcriptions = list(collection.find().sort("timestamp", -1))
        for transcription in transcriptions:
            transcription['_id'] = str(transcription['_id'])
        return jsonify(transcriptions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_transcription/<document_id>', methods=['GET'])
def get_transcription(document_id):
    try:
        document = collection.find_one({"_id": ObjectId(document_id)})
        if document:
            document['_id'] = str(document['_id'])
            return jsonify(document), 200
        return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)