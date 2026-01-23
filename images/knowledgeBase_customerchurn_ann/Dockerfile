FROM busybox:latest

# Safe internal location (NOT /tmp)
WORKDIR /model

# Copy model into image (safe place)
COPY models/customer_churn/ann/currentAiSolution.keras /model/currentAiSolution.keras

# Copy image-specific README (required)
COPY images/knowledgeBase_customerchurn_ann/README.md /

# At CONTAINER START, inject model into shared volume
CMD ["sh", "-c", "mkdir -p /tmp/knowledgeBase && cp /model/currentAiSolution.keras /tmp/knowledgeBase/"]
