# ASL Meeting Application

## Overview
This is the Backend code for a Video Meeting Application, in which user can enable ASL (American Sign Language) Detection functionality (characters recognition from hand gestures).

## Related Repositories
- Frontend: https://github.com/baotram153/ASL-Detection-FE.git

## How to run the Backend code?
1. Create python virtual environment and activate that environment
```sh
git clone https://github.com/baotram153/ASL-Detection.git
cd ASL-Detection
python -m venv env
./env/Scripts/activate
```

2. Install required packages
```sh
pip install -r requirements.txt
```

3. Run the Backend server
```sh
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app
```
- The number of workers should ideally be your number of cpu cores x2+1