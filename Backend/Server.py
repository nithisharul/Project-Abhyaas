import os
import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from together import Together
from gtts import gTTS
from dotenv import load_dotenv
import speech_recognition as sr

# Load env vars
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1")

# Init Together client
client = Together(api_key=TOGETHER_API_KEY)

# FastAPI app
app = FastAPI()

# ----------------------------
# Schemas
# ----------------------------
class QuestionRequest(BaseModel):
    company: str = "General"
    role: str = "Engineer"

class EvalRequest(BaseModel):
    company: str = "General"
    question: str
    answer: str

# ----------------------------
# Company profiles
# ----------------------------
company_profiles = {
    "Google": "Focus on algorithms, system design, and problem-solving depth.",
    "Infosys": "Practical coding, OOP basics, and database concepts.",
    "Microsoft": "System design, product sense, and engineering trade-offs.",
    "TCS": "Practical coding, aptitude questions, and basic problem-solving."
}

# ----------------------------
# Helper functions
# ----------------------------
def capture_voice_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand."
    except sr.RequestError:
        return "Speech recognition service failed."

def ask_together_ai(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message["content"]

def text_to_audio_bytes(text: str):
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf

# ----------------------------
# Endpoint: Ask AI (normal conversation + interview)
# ----------------------------
@app.post("/ask")
async def ask_ai(request: QuestionRequest):
    voice_text = capture_voice_text()
    prompt = f"You are interviewing for {request.role} at {request.company}. Question: {voice_text}"
    ai_reply = ask_together_ai(prompt)
    audio_buf = text_to_audio_bytes(ai_reply)
    return {
        "user_speech": voice_text,
        "ai_reply_text": ai_reply,
        "ai_reply_audio": "/speak"
    }

# ----------------------------
# Endpoint: Stream audio
# ----------------------------
@app.get("/speak")
async def speak():
    sample_text = "This is a demo speech reply."
    audio_buf = text_to_audio_bytes(sample_text)
    return StreamingResponse(audio_buf, media_type="audio/mpeg")

# ----------------------------
# Endpoint: Evaluate candidate answer
# ----------------------------
@app.post("/evaluate-answer")
async def evaluate_answer(req: EvalRequest):
    style = company_profiles.get(req.company, "Balanced.")
    prompt = (
        f"Company: {req.company}, Style: {style}\n"
        f"Question: {req.question}\n"
        f"Answer: {req.answer}\n\n"
        "Evaluate correctness (correct/partial/incorrect), clarity, and suggest improvement. "
        "Return JSON with fields correctness, explanation, improvements."
    )
    ai_eval = ask_together_ai(prompt)
    return JSONResponse({"evaluation": ai_eval})

# ----------------------------
# Endpoint: Generate multiple interview questions
# ----------------------------
@app.post("/generate-questions")
async def generate_questions(req: QuestionRequest):
    style = company_profiles.get(req.company, "Balanced technical + behavioral.")
    prompt = (
        f"Generate 5 interview questions for {req.role} role at {req.company}. "
        f"Company style: {style}. "
        f"Return JSON array with fields id, question, type, hint."
    )
    questions = ask_together_ai(prompt)
    return JSONResponse({"company": req.company, "role": req.role, "questions": questions})

