FROM busybox:latest

# Safe internal directory (NOT /tmp)
WORKDIR /model

# Copy OLS model into image
COPY models/customer_churn/ols/currentOlsSolution.pkl /model/currentOlsSolution.pkl

# Copy image-specific README (required)
COPY images/knowledgeBase_customerchurn_ols/README.md /

# At container START, inject model into shared volume
CMD ["sh", "-c", "mkdir -p /tmp/knowledgeBase && cp /model/currentOlsSolution.pkl /tmp/knowledgeBase/"]
