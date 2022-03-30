FROM python:3.9
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./main.py /code/main.py
COPY ./ppie_rf.joblib /code/ppie_rf.joblib
COPY ./ppie_rf_bad.joblib /code/ppie_rf_bad.joblib
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]