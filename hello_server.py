from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def home():
    return {"message":"Hello! I am a server"}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/echo")
def echo(data:dict):
    return {"you_Sent":data}


