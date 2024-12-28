from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from together import Together
import speech_recognition as sr
from pydub import AudioSegment

from dotenv import load_dotenv

import openai
import os
import json
import requests

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

CHATS = 'database.json'


origins = [
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/" , response_class=HTMLResponse)
async def root(request: Request):
    # return {"message": "Welcome to Interviewer bot"}
    messages = load_messages()
    return templates.TemplateResponse("index.html", {"request": request, "messages": messages})

@app.post("/talk")
async def post_audio(file: UploadFile):
    # send audio file to openai whisper and get it transcribed
    user_message = transcribe_audio(file)
    print(f"\n=====\nUser message: {user_message}\n=====\n")

    # get chat response from openai
    chat_response = get_chat_response(user_message)
    if not chat_response:
        return {"error": "Failed to get chat response"}
    
    # convert chat response to audio
    audio_output = text_to_speech(chat_response)

    if not audio_output:
        return {"error": "Failed to generate audio response"}
    
    def iterfile():
        yield audio_output

    return StreamingResponse(iterfile(), media_type="application/octet-stream")

@app.get("/clear")
async def clear_history():
    file = 'database.json'
    open(file,'w')
    print("\nChat History cleared\n")
    # return {"message": "Chat History cleared"}
    return RedirectResponse(url="/")


#functions
# ! this works when I tested it with a file
# def transcribe_audio(file):
#     filepath = r"E:\Work\new-ai-interviewer\backend\recording.wav"
#     converted_path = r"E:\Work\new-ai-interviewer\backend\converted.wav"
    
#     # Convert with pydub first
#     sound = AudioSegment.from_file(filepath)
#     sound = sound.set_frame_rate(16000).set_channels(1)
#     sound.export(converted_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    
#     recognizer = sr.Recognizer()
    
#     try:
#         with sr.AudioFile(converted_path) as source:
#             audio_data = recognizer.record(source)
#             transcript = recognizer.recognize_google(audio_data)
#             return {"text": transcript}
#     except sr.UnknownValueError:
#         return {"text": "Could not understand audio"}
#     except sr.RequestError:
#         return {"text": "Network error"}
#     # finally:
#     #     if os.path.exists(converted_path):
#     #         os.remove(converted_path)


def transcribe_audio(file):
    # Write uploaded file to temp WAV
    with open("temp.wav", "wb") as buffer:
        buffer.write(file.file.read())
    
    # Convert format for speech recognition
    sound = AudioSegment.from_file("temp.wav")
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export("converted.wav", format="wav", parameters=["-acodec", "pcm_s16le"])
    
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile("converted.wav") as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
            return {"text": transcript}
    except sr.UnknownValueError:
        return {"text": "Could not understand audio"}
    except sr.RequestError:
        return {"text": "Network error"}
    finally:
        # Cleanup temp files
        for f in ["temp.wav", "converted.wav"]:
            if os.path.exists(f):
                os.remove(f)

def get_chat_response(user_message):
    messages = load_messages()
    messages.append({"role": "user", "content": user_message['text']})

    # Send to ChatGpt/OpenAi
    # gpt_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages
    #     )
    # print(f"GPT Response: {gpt_response}")
    
    client = Together()

    gpt_response = client.chat.completions.create(
        model="Qwen/QwQ-32B-Preview",
        messages=messages,
    )
    # !
    # print()
    # print("GPT Response: ", gpt_response)
    # print()
    # parsed_gpt_response = gpt_response['choices'][0]['message']['content']
    parsed_gpt_response = gpt_response.choices[0].message.content
    print("++++++++++++++")
    print(f"Response from Qwen: {parsed_gpt_response}")
    print("++++++++++++++")

    text_to_speech(parsed_gpt_response)

    # Save messages
    save_messages(user_message['text'], parsed_gpt_response)

    return parsed_gpt_response

def load_messages():
    messages=[]
    file = 'database.json'

    empty = os.stat(file).st_size == 0

    if not empty:
        with open(file, 'r') as db_file:
            data = json.load(db_file)
            for item in data:
                messages.append(item)
    else:
        messages.append(
            {"role":"system","content": "You are interviewing the user for a AI/ML developer position. Ask short questions that are relevant to a junior level developer. Your name is Kuttan. The user is Achu. Keep responses under 30 words and be funny sometimes. Remember to ask one question at a time if it is a technical question, then follow up on that if you like."}
        )

    print(f"Messages loaded ")
    print(messages)
    return messages

def save_messages(user_message, gpt_response):
    file = 'database.json'
    messages = load_messages()

    messages.append({"role":"user","content": user_message})
    messages.append({"role":"assistant","content": gpt_response})
    with open(file, 'w') as db_file:
        json.dump(messages, db_file)

def text_to_speech(text):
    voice_id = 'pNInz6obpgDQGcFmaJgB'
    
    body = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }

    headers = {
        "Content-Type": "application/json",
        "accept": "audio/mpeg",
        "xi-api-key": elevenlabs_api_key
    }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    try:
        response = requests.post(url, json=body, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses
        if response.status_code == 200:
            return response.content
        else:
            print('something went wrong')
    except requests.exceptions.RequestException as e:
        print(f"Text-to-speech API call failed: {e}")
        return None
    except Exception as e:
        print(f"Error Occured: {e}")

