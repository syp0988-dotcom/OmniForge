@echo off
cd /d G:\multi_agent
.venv\Scripts\python.exe -m uvicorn agentflow.app.main:app --host 0.0.0.0 --port 8000 > G:\multi_agent\backend.log 2>&1
