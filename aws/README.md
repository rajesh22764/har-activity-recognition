# AWS deployment notes

**Status: artifacts are prepared but were NOT deployed or tested on AWS** (no AWS account was used).

| Need | Artifact | AWS service |
|---|---|---|
| Real-time model endpoint | `inference.py` (SageMaker SKLearn contract) | SageMaker endpoint, model artifact in S3 |
| Dashboard | `Dockerfile` (build from project root: `docker build -f aws/Dockerfile .`) | ECR + App Runner / ECS Fargate |
| Batch scoring of sensor files | `src/` functions + the saved model | S3 trigger -> Lambda / SageMaker Batch Transform |
| Data lake | `data/raw` | S3 (+ Athena for ad-hoc SQL over feature tables) |

Suggested path: upload `models/` to S3 -> create a SageMaker `SKLearnModel` with `entry_point=inference.py`
-> `deploy(instance_type="ml.t2.medium")` -> call with `{"features": [[...561 floats...]]}`.
