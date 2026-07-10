import os, json, httpx
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-3.5-flash"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DebateState(BaseModel):
    phase: str
    issue: str
    history: List[Dict[str, str]]

async def send_discord_error(api_name: str, error_message: str):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️ [Discord] DISCORD_WEBHOOK_URL 환경변수가 세팅되지 않아 알림을 건너뜁니다.")
        return
    
    payload = {
        "embeds": [{
            "title": f"🚨 백엔드 에러 발생 ({api_name})",
            "description": f"**상세 에러 내용:**\n```{error_message}```",
            "color": 15158332,
            "footer": {"text": "LogiNews 실시간 시스템 모니터링"}
        }]
    }
    
    async with httpx.AsyncClient() as client_http:
        try:
            await client_http.post(webhook_url, json=payload)
        except Exception as e:
            print(f"❌ 디스코드 웹후크 발송 실패: {e}")

@app.get("/")
async def root():
    return {"message": "LogiNews API is running"}

@app.get("/api/search-news")
async def search_news(query: str = "속보"):
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
    }
    async with httpx.AsyncClient() as client_http:
        try:
            res = await client_http.get(
                "https://openapi.naver.com/v1/search/news.json",
                params={"query": query, "display": 9, "sort": "sim"},
                headers=headers,
            )
            if res.status_code != 200:
                error_info = f"Status Code: {res.status_code}\nResponse: {res.text}"
                await send_discord_error("네이버 뉴스 검색 API", error_info)
                raise HTTPException(status_code=res.status_code, detail="Naver API Error")
            return res.json()
        except Exception as e:
            if not isinstance(e, HTTPException):
                await send_discord_error("네이버 뉴스 검색 API (네트워크/코드)", str(e))
                raise HTTPException(status_code=500, detail=str(e))
            raise e

@app.post("/api/analyze-news")
async def analyze(request: Dict):
    prompt = f"뉴스 내용: {request['content']}\n핵심 쟁점 1개를 추출해줘. JSON: {{'issue1': '...'}}"
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(res.text)
    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API Error (analyze-news): {error_msg}")

        await send_discord_error("뉴스 분석 API (Gemini)", error_msg)
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {error_msg}")

@app.post("/api/debate-step")
async def debate_step(state: DebateState):
    prompts = {"입론": "반박", "반론": "공격", "재반론": "방어", "최종결론": "정리"}
    if state.phase not in prompts:
        raise HTTPException(status_code=400, detail=f"Unknown phase: {state.phase}")
    
    system_instruction = (
        f"당신은 찬반 토론을 진행하는 상대방 토론자입니다. 당신의 목표는 사용자의 의견에 동조하거나 비난하는 것이 아니라, 논리적 주장과 반박을 제시하는 것입니다.\n"
        f"현재 토론 단계: [{state.phase}]\n"
        f"토론 주제: {state.issue}\n"
        f"당신의 전략: {prompts[state.phase]}\n"
        f"상대방(User)의 말을 논리적으로 반박하거나 의견을 정리하여 마크다운 문법을 사용하지 않은 한국어 텍스트 형식으로 간결하고 명확하게 답변하세요. 비속어를 사용하지 말고, 격식체를 유지하며 어떠한 경우에도 상대방을 비난하지 마세요."
    )

    if not state.history:
        raise HTTPException(status_code=400, detail="History cannot be empty")

    last_user_message = state.history[-1]['content']

    formatted_history = []
    for m in state.history[:-1]:
        role = "user" if m["role"] == "user" else "model"
        formatted_history.append(
            types.Content(role=role, parts=[types.Part(text=m["content"])])
        )

    try:
        chat = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            history=formatted_history,
        )

        res = chat.send_message(last_user_message)
        return {"reply": res.text}

    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API Error (debate-step): {error_msg}")
        
        await send_discord_error("토론 단계 API (Gemini)", error_msg)
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {error_msg}")
