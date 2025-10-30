FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

EXPOSE 8501
CMD ["streamlit", "run", "rct_field_flow/monitor.py", "--server.port=8501", "--server.address=0.0.0.0"]