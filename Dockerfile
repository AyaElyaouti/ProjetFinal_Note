FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install pandas scikit-learn
CMD ["python", "evaluation.py"]
