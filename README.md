# cbo-fl

**Contextual Bayesian Optimization in Resource-Aware Federated Learning for Vehicular Applications**

A federated learning (FL) system where each client uses **Contextual Bayesian Optimization (CBO)** to automatically tune its local training hyperparameters (batch size, epochs, numeric precision) while respecting resource constraints (CPU, memory, training time, and model size).

---

## Architecture

```
cbo-fl/
├── server/
│   ├── server.py          # Flower server (multi-task FedAvg)
│   ├── Dockerfile
│   └── requirements.txt
├── client/
│   ├── client.py          # Flower client + CBO
│   ├── utils.py           # Models, training functions, data partitioning
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml     # Container orchestration
├── requirements.txt       # Dependencies for local execution
└── utils.py               # Local version of utils
```

The server coordinates **3 tasks** sequentially (MLP models with distinct architectures). Each round, the server instructs clients with the current task and aggregates parameters via FedAvg.

---

## Prerequisites

### Local execution
- Python **3.12**
- pip
- *(Optional)* CUDA 12.4 + compatible NVIDIA driver for GPU training

### Docker execution
- Docker **24+**
- Docker Compose **v2** (`docker compose` — no hyphen)

---

## Local Execution

### 1. Install dependencies

> Run from the project root.

**With GPU (CUDA 12.4):**
```bash
pip install -r requirements.txt
pip install flwr==1.15.2
```

**CPU only:**
```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flwr==1.15.2 torchao==0.12.0 torchmetrics==1.8.2 scikit-learn==1.8.0 \
            tqdm==4.67.1 psutil==7.2.1 matplotlib==3.10.8 ptflops==0.7.5 pynvml==13.0.1
```

### 2. Create output directories

```bash
mkdir -p client/figure client/data server/figure
```

### 3. Start the server

Open a terminal and run from the `server/` directory:

```bash
cd server
python server.py
```

The server will wait for incoming connections on `0.0.0.0:8080`.

### 4. Start the clients

Open **one separate terminal per client** (minimum 2, maximum 4) and run from the `client/` directory:

```bash
cd client

# Terminal 1
python client.py --partition-id 0

# Terminal 2
python client.py --partition-id 1

# Terminal 3 (optional)
python client.py --partition-id 2

# Terminal 4 (optional)
python client.py --partition-id 3
```

> **Note:** The server requires at least **2 connected clients** before starting training. The `--partition-id` flag determines which IID slice of the California Housing dataset that client will use.

### 5. Outputs

| Output                   | Location                                    |
| ------------------------ | ------------------------------------------- |
| Resource usage charts    | `client/figure/`                            |
| Training history (CSV)   | `client/data/training_results.csv`          |
| Validation history (CSV) | `client/data/validation_results.csv`        |
| Global loss curve        | `server/figure/federated_learning_loss.png` |

---

## Docker Execution

### 1. Create output directories on the host

```bash
mkdir -p client/figure client/data server/figure
```

### 2. Build and start

From the **project root**:

```bash
docker compose up --build
```

This will:
1. Build the server and client images
2. Start the `vf54_fl_server` container (Flower server on port 8080)
3. Start containers `vf54_fl_client_0` through `vf54_fl_client_3`, each with its own data partition

### 3. Start without rebuilding (after the first build)

```bash
docker compose up
```

## Key Configuration (`utils.py`)

| Parameter           | Value          | Description                         |
| ------------------- | -------------- | ----------------------------------- |
| `N_CLIENTS`         | `4`            | Number of federated clients         |
| `ROUNDS`            | `150`          | Federated training rounds           |
| `PROC_MAX`          | `35%`          | Maximum CPU usage per client        |
| `MEM_MAX`           | `60 MB`        | Maximum RAM usage per client        |
| `TRAINING_TIME_MAX` | `70 s`         | Maximum training time per trial     |
| `MODEL_SIZE_LIMIT`  | `5 MB`         | Maximum serialized model size       |
| `DEVICE`            | `cpu`          | Training device (`cpu` or `cuda`)   |
| `BATCH_SIZE`        | 128–1024       | Batch size search space (log2 grid) |
| `EPOCHS`            | 1–20           | Epoch search space                  |
| Precision           | 8, 16, 32 bits | Numeric precision search space      |
