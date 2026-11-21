unzip clausevault-us-health-python.zip
cd clausevault-us-health-python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 test_pipeline.py
uvicorn main:app --reload --port 8000