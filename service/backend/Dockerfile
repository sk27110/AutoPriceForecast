FROM pingvan/car-price-prediction-python:latest

WORKDIR /app/service/backend

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /app

EXPOSE 8000

CMD ["uvicorn", "service.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]