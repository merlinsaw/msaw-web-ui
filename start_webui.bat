@echo off
echo Setting up environment...
set OPENAI_API_KEY=your-api-key-here
set OPENAI_MODEL=gpt-4
echo Starting Web UI on http://127.0.0.1:7788
python webui.py --ip 127.0.0.1 --port 7788
pause
