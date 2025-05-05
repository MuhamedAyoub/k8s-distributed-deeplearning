#  Distributed Deep Learning on Kubernetes: Horovod + MPI Operator + Loki/Grafana

This project provides a production-ready setup for running **distributed deep learning training jobs** using **Horovod with MPI** on Kubernetes, and observability using **Grafana Loki Stack**.

---

## 📦 Stack Components

| Component           | Role                                                  |
|---------------------|-------------------------------------------------------|
| **Grafana Loki**     | Log aggregation and visualization                    |
| **Promtail**         | Log shipper that sends logs to Loki                  |
| **Grafana**          | Dashboard UI for visualizing logs and metrics        |
| **MPI Operator**     | Manages distributed MPI workloads using CRDs         |
| **Horovod Job**      | Distributed TensorFlow training using MPI and Horovod|

---

## Quick Start

###  Prerequisites

- A working Kubernetes cluster (`k3s`, `kind`, or cloud)
- `kubectl` installed and configured
- `helm` installed
- Docker installed and running

---

### Setup Instructions

1. **Build Horovod Training Image**

```bash
docker build -t horovod:latest ./horovod
```


Your ./horovod folder should include a Dockerfile and tensorflow_mnist.py.

2. **Run the Deployment Script**
```bash
chmod +x deploy_stack.sh
./deploy_stack.sh
```

#### This script will
- Create required namespaces

- Install Loki + Grafana via Helm

- Install the MPI Operator (v2beta1)

- Deploy a distributed TensorFlow Horovod job using the MPIJob CRD