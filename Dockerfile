FROM python:latest
FROM pytorch/pytorch:latest

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "./run_service.py"]