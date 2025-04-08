Run: python -m PyInstaller -y keystroke.py to from this directory to build the executable

runs the api
python -m uvicorn IFApi:app --reload


runs apiv2
python -m uvicorn IFApiV2:app --reload