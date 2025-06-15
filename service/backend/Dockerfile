FROM pingvan/car-price-prediction-backend:latest

WORKDIR /app/service/backend

COPY . ./

ENV PYTHONPATH /app

EXPOSE 8000

CMD ["uvicorn", "service.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]