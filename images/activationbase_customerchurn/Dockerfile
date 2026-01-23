FROM busybox:latest

# Safe internal directory
WORKDIR /data

# Copy activation data into image (safe place)
COPY data/customer-churn-dataset/processed/activation_data.csv /data/activation_data.csv

# Copy image-specific README (required)
COPY images/activationBase_customerchurn/README.md /

# At CONTAINER START, inject data into shared volume
CMD ["sh", "-c", "mkdir -p /tmp/activationBase && cp /data/activation_data.csv /tmp/activationBase/activation_data.csv"]
