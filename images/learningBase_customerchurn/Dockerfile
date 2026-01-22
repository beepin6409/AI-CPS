FROM busybox:latest

RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

COPY data/customer-churn-dataset/processed/training_data.csv /tmp/learningBase/train/training_data.csv
COPY data/customer-churn-dataset/processed/test_data.csv /tmp/learningBase/validation/test_data.csv

COPY images/learningBase_customerchurn/README.md /

