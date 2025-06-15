FROM car-price-prediction-deps:latest

WORKDIR /app

COPY . /app/src/server

RUN pip install python-multipart

ENV PYTHONPATH /app

EXPOSE 8000

CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8000"]