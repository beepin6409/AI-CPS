FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    tensorflow \
    statsmodels \
    scikit-learn

COPY images/codeBase_customerchurn/run_inference.py .

COPY images/codeBase_customerchurn/README.md /

CMD ["python", "run_inference.py"]
