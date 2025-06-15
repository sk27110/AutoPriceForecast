FROM car-price-prediction-deps:latest

WORKDIR /app/steamlit

COPY . /app/steamlit

# RUN pip install streamlit
RUN python3 -m pip install --no-cache-dir streamlit

RUN ls -la

EXPOSE 8501

CMD ["python3", "-c", "import os; print(f'Current dir: {os.getcwd()}'); print(os.listdir('.')); print('main.py exists: ' + str(os.path.exists('main.py')))"]
