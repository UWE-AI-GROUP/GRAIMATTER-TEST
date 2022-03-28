# PPIE Deployed model example

## Introduction

This folder contains the code needed to build and deploy a toy model. The model predicts risk of a stroke from four inputs:
- `age` (integer)
- `blood_pressure` (integer)
- `smoker`
- `total_cholestorol`

`generate_model.py` includes a notebook that generates data (with a very strong signal) and fits a random forest. A trained model is provided (`ppie_rf.joblib`).

## Running locally

If you have the requirements installed in whatever environment you have setup (see `requirements.txt`), you can run the `fastapi` server locally for testing. From `GRAIMatter/ppie_example` run:
```bash
uvicorn main:app --reload
```
(the `--reload` flag means that the app will automatically reload if the code is changed). You can test this by pointing your browser at:
```
http://127.0.0.1:8000/predict/?age=60&blood_pressure=110&smoker=0&total_cholestorol=5
```
and you should receive an output in the browser that looks something like:
```
{'stroke_risk' : 0.8}
```
You can also submit a request via curl as either a GET:
```bash
curl 'http://127.0.0.1:8000/predict/?age=60&blood_pressure=110&smoker=0&total_cholestorol=5'
```
or POST:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/post_predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 70,
  "blood_pressure": 100,
  "smoker": 1,
  "total_cholestorol": 5
}'
```
Note that fastapi creates great doc pages that give you these URLs: http://127.0.0.1/docs

## Docker

Build the docker image (from `GRAIMatter/ppie_example`) with:
```bash
docker build -t ppie-example:latest -f ./Dockerfile .
```
The Dockerfile is very basic, and may require changes for a deployment.

Once the docker image has built, you can run it with:
```bash
docker run -it -p 80:80 ppie-example:latest
```
You should be able to access the model exactly as in the previous examples, but changing the port from 8000 to 80.