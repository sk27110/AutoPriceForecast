FROM pingvan/car-price-prediction-python:latest

WORKDIR /app/service/frontend

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /app

EXPOSE 8501

CMD ["streamlit", "run", "service/frontend/main.py", "--server.port=8501", "--server.address=0.0.0.0"]