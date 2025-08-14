FROM python:3.9-slim

RUN pip uninstall -y pydantic || true
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --force-reinstall "pydantic>=2.5.0"
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]