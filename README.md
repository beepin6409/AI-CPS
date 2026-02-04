# AI-Based Customer Churn Prediction System for Subscription Services
**Dockerized ANN & OLS Application using the AI-CPS Architecture**

## Authors / Owners
- **Bipin Ghimire**
- **Paul Bakos**

---

## Course Context

This project was developed as part of the course:

**M. Grum: Advanced AI-based Application Systems**  
Junior Chair for Business Information Systems,  
esp. AI-based Application Systems  
**University of Potsdam**

---

## Project Overview

This repository implements an **AI-Based Customer Churn Prediction System for Subscription Services** following the architectural principles of the **AI-CPS (AI-based Cyber-Physical Systems) platform**.

The system predicts whether a customer is likely to **churn (cancel a subscription)** using two complementary modeling approaches:

- **Artificial Neural Network (ANN)** for non-linear pattern learning  
- **Ordinary Least Squares (OLS)** for interpretable statistical modeling  

The project demonstrates how **AI knowledge, activation data, and execution logic** can be modularized, containerized, and orchestrated using Docker, enabling:

- Node-independent deployment  
- Clear separation of concerns  
- Reproducible AI application scenarios  

Both models are trained offline and applied to the same activation dataset using Docker Compose–based execution scenarios.

---

## AI-CPS Conceptual Mapping

The project strictly follows the AI-CPS paradigm by separating AI system components:

| AI-CPS Component | Purpose | Implementation |
|------------------|--------|----------------|
| KnowledgeBase | Stores trained AI models | ANN (`.keras`), OLS (`.pkl`) |
| ActivationBase | Stores activation data | `activation_data.csv` |
| CodeBase | Executes inference logic | `run_inference.py` |
| LearningBase | Training material | CSV datasets (offline only) |

All runtime interaction between components is realized using a **shared Docker volume** (`ai_system`) mounted to `/tmp`.


## Repository Structure



```text
AI-CPS/
│
├── code/
│   └── customer_churn_prediction/
│       ├── data_scraping.py
│       ├── data_preprocessing.py
│       ├── ann_model.py
│       ├── ols_model.py
│       └── diagnostic_plots.py
│
├── data/
│   └── customer-churn-dataset/
│       ├── raw/
│       │   └── customer_churn.csv
│       └── processed/
│           ├── training_data.csv
│           ├── test_data.csv
│           ├── activation_data.csv
│           └── joint_collection.csv
│
├── documentation/
│   └── customer_churn/
│       ├── ann/
│       └── ols/
│
├── models/
│   └── customer_churn/
│       ├── ann/
│       │   └── currentAiSolution.keras
│       └── ols/
│           └── currentOlsSolution.pkl
│
├── images/
│       ├── knowledgeBase_customerchurn_ann/
│       │   ├── Dockerfile
│       │   └── README.md
│       │
│       ├── knowledgeBase_customerchurn_ols/
│       │   ├── Dockerfile
│       │   └── README.md
│       │
│       ├── activationBase_customerchurn/
│       │   ├── Dockerfile
│       │   └── README.md
│       │
│       └── codeBase_customerchurn/
│           ├── Dockerfile
│           ├── README.md
│           └── run_inference.py
│
├── scenarios/
│       ├── apply_ann_customerchurn/
│       │   └── docker-compose.yml
│       └── apply_ols_customerchurn/
│           └── docker-compose.yml
│
└── README.md


```


## Docker Images (Public)

All Docker images are publicly available on Docker Hub.

**Docker Hub ID:** `beepin6409`

### KnowledgeBase Images
- `beepin6409/knowledgebase_customer_churn_ann`
- `beepin6409/knowledgebase_customer_churn_ols`

### ActivationBase Image
- `beepin6409/activationbase_customer_churn`

### CodeBase Image
- `beepin6409/codebase_customer_churn`

Each image:
- Is self-contained
- Uses a minimal base image (BusyBox for data containers)
- Includes an image-specific `README.md`
- Declares ownership, course context, and license
- Complies with **AGPL-3.0**

---

## Docker Pull Commands

```bash
docker pull beepin6409/knowledgebase_customer_churn_ann
docker pull beepin6409/knowledgebase_customer_churn_ols
docker pull beepin6409/activationbase_customer_churn
docker pull beepin6409/codebase_customer_churn
```
Running the ANN Inference Scenario
```bash
docker volume create ai_system

docker-compose -f scenarios/apply_ann_customerchurn/docker-compose.yml up
```
Expected Output:

```bash
ANN predictions: [0.18...]
```
Running the OLS Inference Scenario
```bash
docker volume create ai_system

docker-compose -f scenarios/apply_ols_customerchurn/docker-compose.yml up
```
Expected Output:
```bash

OLS predictions: [0.23...]
```
## Technical Stack
Python 3.10

TensorFlow / Keras

Statsmodels

Scikit-learn

Docker & Docker Compose

BusyBox

The system is platform-independent and executable on any Docker-supported environment.

## Requirements

### Hardware Requirements
- x86_64 compatible system
- Sufficient disk space for Docker images and volumes
- Not compatible with ARM64 architectures (e.g. Apple Silicon / M1 / M2 / M3)

### Software Requirements
- Docker (version 20.10 or newer)
- Docker Compose (v2 or newer)

Note:
This project was tested only on x86_64 systems.  
ARM-based systems (such as Apple Silicon) are currently not supported due to
library and Docker image compatibility issues.

## Reproducibility
The project is fully reproducible using:

Public GitHub repository

Public Docker Hub images

Version-controlled Docker Compose scenarios

No local model training or configuration is required to run inference.

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
