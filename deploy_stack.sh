#!/bin/bash

set -e

# --------------------------
# CONFIGURATION VARIABLES
# --------------------------
NAMESPACE=ml-ops
LOKI_NAMESPACE=loki
MPI_NAMESPACE=mpi-operator

echo "🔧 Creating namespaces..."
kubectl create namespace $NAMESPACE || true
kubectl create namespace $LOKI_NAMESPACE || true
kubectl create namespace $MPI_NAMESPACE || true

# --------------------------
# INSTALL GRAFANA LOKI STACK
# --------------------------
echo "📦 Installing Grafana Loki Stack..."

helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm upgrade --install loki grafana/loki-stack \
  --namespace $LOKI_NAMESPACE \
  --set grafana.enabled=true \
  --set promtail.enabled=true \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=5Gi \
  --wait

# --------------------------
# INSTALL MPI OPERATOR
# --------------------------
echo "📦 Installing MPI Operator..."

kubectl apply --server-side -f https://raw.githubusercontent.com/kubeflow/mpi-operator/master/deploy/v2beta1/mpi-operator.yaml


# --------------------------
# DEPLOY HOROVOD MPI JOB
# --------------------------
echo "🚀 Deploying Horovod Job..."

kubectl apply -n $NAMESPACE -f - <<EOF
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: tensorflow-mnist
spec:
  slotsPerWorker: 1
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - image: docker.io/kubeflow/mpi-horovod-mnist
            name: mpi-launcher
            command:
            - mpirun
            args:
            - -np
            - "2"
            - --allow-run-as-root
            - -bind-to
            - none
            - -map-by
            - slot
            - -x
            - LD_LIBRARY_PATH
            - -x
            - PATH
            - -mca
            - pml
            - ob1
            - -mca
            - btl
            - ^openib
            - python
            - /examples/tensorflow_mnist.py
            resources:
              limits:
                cpu: 1
                memory: 2Gi
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - image: docker.io/kubeflow/mpi-horovod-mnist
            name: mpi-worker
            resources:
              limits:
                cpu: 2
                memory: 4Gi

EOF

echo "✅ All components deployed."
