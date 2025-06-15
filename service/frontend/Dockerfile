FROM pingvan/car-price-prediction-frontend:latest

WORKDIR /app/service/frontend

COPY . ./

ENV PYTHONPATH /app

EXPOSE 8501

CMD ["streamlit", "run", "service/frontend/main.py", "--server.port=8501", "--server.address=0.0.0.0"]