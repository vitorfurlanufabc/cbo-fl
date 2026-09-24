# %%
# imports
import os, sys, logging, warnings

os.environ["PYTORCH_CPP_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
sys.stderr = open(os.devnull, "w")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader, Subset
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
import torch.ao.quantization as tq
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from multiprocessing import Process, Manager
from scipy.stats import norm
from ptflops import get_model_complexity_info
import time, psutil, threading, subprocess, math, re, copy, pynvml, io, csv
import numpy as np, matplotlib.pyplot as plt


# %%
# variables
HIDDEN_UNITS = [128]
EPOCHS = 20
BATCH_SIZE = 16
PRECISION = 8

# GP (RBF)
SIGMA_F2 = 1.0
ELL = 2.0
SIGMA_N2 = 1e-6

# Search space
BATCH_SIZE_MIN = 128
BATCH_SIZE_MAX = 1024
LOG2_BATCH_SIZE_MIN = math.log2(BATCH_SIZE_MIN)
LOG2_BATCH_SIZE_MAX = math.log2(BATCH_SIZE_MAX)
EPOCHS_MIN = 1
EPOCHS_MAX = 10
# PRECISION_MIN = 4
PRECISION_MIN = 8
PRECISION_MAX = 32
LOG2_PRECISION_MIN = math.log2(PRECISION_MIN)
LOG2_PRECISION_MAX = math.log2(PRECISION_MAX)

LOG2_BATCH_SIZE_GRID = np.arange(LOG2_BATCH_SIZE_MIN, LOG2_BATCH_SIZE_MAX + 1e-12, 0.25)
EPOCHS_GRID = np.arange(EPOCHS_MIN, EPOCHS_MAX + 1e-12, 1)
LOG2_PRECISION_GRID = np.arange(LOG2_PRECISION_MIN, LOG2_PRECISION_MAX + 1e-12, 1)
TASKS = [1, 2, 3]
GRID = [
    (log2_batch_size, epochs, log2_precision)
    for log2_batch_size in LOG2_BATCH_SIZE_GRID
    for epochs in EPOCHS_GRID
    for log2_precision in LOG2_PRECISION_GRID
]

# restrições e penalidades
# DEVICE = "cpu"
DEVICE = "cuda"

PROC_MAX = 5
MEM_MAX = 25
TRAINING_TIME_MAX = 5
MODEL_SIZE_LIMIT = 10

LAMBDA_LOSS = 2
LAMBDA_PROC = 1
LAMBDA_MEM = 1
LAMBDA_TIME = 1
LAMBDA_MODEL_SIZE = 1

NUM_ITERS = 2


# %%
# classes and functions
class MLP(nn.Module):
    """
    Parametric Multi-Layer Perceptron (MLP) architecture.
    """

    def __init__(
        self,
        input_size=3 * 32 * 32,
        num_classes=10,
        hidden_units=[512, 256, 128],
        dropout_rate=0.3,
    ):
        super(MLP, self).__init__()

        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        layers = []
        in_features = input_size

        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = units

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


class GPUSampler:
    """
    GPU sampler for monitoring GPU usage.
    """

    def __init__(self):
        self.available = False
        self.is_jetson = False

        self.usage_samples = []
        self.mem_samples = []
        self.power_samples = []
        self.temp_samples = []

        try:
            self.nvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
            self.backend = "nvml"
        except:
            if os.path.exists("/usr/bin/tegrastats"):
                self.is_jetson = True
                self.available = True
                self.backend = "tegrastats"
            else:
                self.backend = "none"

        self.proc = None

    def sample_nvml(self):
        try:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            meminfo = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used = meminfo.used
            power = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW → W
            temp = self.nvml.nvmlDeviceGetTemperature(
                self.handle, self.nvml.NVML_TEMPERATURE_GPU
            )

            self.usage_samples.append(util)
            self.mem_samples.append(mem_used)
            self.power_samples.append(power)
            self.temp_samples.append(temp)

        except:
            pass

    def start_tegrastats(self):
        self.proc = subprocess.Popen(
            ["/usr/bin/tegrastats", "--interval", "50"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def sample_tegrastats(self):
        if self.proc is None:
            return
        try:
            line = self.proc.stdout.readline().strip()

            util_match = re.search(r"GR3D_FREQ (\d+)%", line)
            util = int(util_match.group(1)) if util_match else 0

            mem_used = 0

            temp_match = re.search(r"GPU@(\d+)C", line)
            temp = int(temp_match.group(1)) if temp_match else 0

            self.usage_samples.append(util)
            self.mem_samples.append(mem_used)
            self.temp_samples.append(temp)

        except:
            pass

    def sample(self):
        if not self.available:
            return
        if self.backend == "nvml":
            self.sample_nvml()
        elif self.backend == "tegrastats":
            self.sample_tegrastats()

    def close(self):
        if self.backend == "nvml":
            try:
                self.nvml.nvmlShutdown()
            except:
                pass
        if self.proc:
            self.proc.kill()


class MetricSampler(threading.Thread):
    """
    Metric sampler for monitoring CPU, RAM, VMS, and GPU usage.
    """

    def __init__(self, interval=0.05):
        super().__init__()
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_flag = False

        # CPU
        self.cpu_usage_accum = 0.0
        self.cpu_samples = 0
        self.cpu_peak = 0.0

        # RAM
        self.ram_samples = []
        self.ram_peak = 0
        self.ram_usage = 0

        # VMS
        self.vms_samples = []
        self.vms_peak = 0
        self.vms_usage = 0

        # GPU
        self.gpu = GPUSampler()
        if self.gpu.backend == "tegrastats":
            self.gpu.start_tegrastats()

        self.process.cpu_percent(interval=None)

    def run(self):
        while not self.stop_flag:
            # RAM
            mem_info = self.process.memory_full_info()
            uss = mem_info.uss
            self.ram_usage = uss
            self.ram_peak = max(self.ram_peak, uss)
            self.ram_samples.append(uss)

            # VMS
            vms = mem_info.vms
            self.vms_usage = vms
            self.vms_peak = max(self.vms_peak, vms)
            self.vms_samples.append(vms)

            # CPU
            cpu_now = self.process.cpu_percent(interval=None)
            self.cpu_usage_accum += cpu_now
            self.cpu_samples += 1
            self.cpu_peak = max(self.cpu_peak, cpu_now)

            # GPU
            self.gpu.sample()

            time.sleep(self.interval)

    def stop(self):
        self.stop_flag = True
        self.join()
        self.gpu.close()

    def get_cpu_average(self):
        return (self.cpu_usage_accum / self.cpu_samples) if self.cpu_samples else 0

    def get_cpu_integral(self):
        return self.cpu_usage_accum

    def get_ram_mean(self):
        return np.mean(self.ram_samples) if self.ram_samples else 0

    def get_ram_integral(self):
        return sum(self.ram_samples) if self.ram_samples else 0

    def get_ram_std(self):
        return np.std(self.ram_samples) if self.ram_samples else 0

    def get_ram_median(self):
        return np.median(self.ram_samples) if self.ram_samples else 0

    def get_ram_max(self):
        return max(self.ram_samples) if self.ram_samples else 0

    def get_vms_mean(self):
        return np.mean(self.vms_samples) if self.vms_samples else 0

    def get_vms_std(self):
        return np.std(self.vms_samples) if self.vms_samples else 0

    def get_vms_median(self):
        return np.median(self.vms_samples) if self.vms_samples else 0

    def get_vms_max(self):
        return max(self.vms_samples) if self.vms_samples else 0


class Monitor:
    """
    Training monitor to collect various performance metrics.
    """

    def __init__(
        self,
        model=None,
        log_per_batch=False,
        device="gpu",
        input_shape=None,
        sample_interval=0.05,
    ):
        self.model = model
        self.log_per_batch = log_per_batch
        self.device = device
        self.input_shape = input_shape
        self.sample_interval = sample_interval
        self.start_time = None

        self.num_params = self._compute_num_params() if model else None
        self.flops_forward = (
            self._compute_flops_forward() if model and input_shape else None
        )

        self.sampler = None

    def _compute_num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def _compute_flops_forward(self):

        macs, _ = get_model_complexity_info(
            self.model,
            self.input_shape,
            as_strings=False,
            print_per_layer_stat=False,
        )
        return macs * 2

    def start(self):
        self.baseline_mem = psutil.Process(os.getpid()).memory_full_info().uss
        self.start_time = time.time()
        self.sampler = MetricSampler(interval=self.sample_interval)
        self.sampler.start()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def end(self):
        self.sampler.stop()
        elapsed = time.time() - self.start_time

        flops_backward = self.flops_forward * 2 if self.flops_forward else None
        flops_total = (
            self.flops_forward + flops_backward if self.flops_forward else None
        )
        flops_per_sec = flops_total / elapsed if flops_total else None

        num_cores = psutil.cpu_count(logical=True)

        gpu_usage = (
            np.mean(self.sampler.gpu.usage_samples)
            if self.sampler.gpu.usage_samples
            else 0
        )
        gpu_peak = (
            max(self.sampler.gpu.usage_samples) if self.sampler.gpu.usage_samples else 0
        )
        gpu_peak_bytes = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )

        gpu_mem = (
            np.mean(self.sampler.gpu.mem_samples) if self.sampler.gpu.mem_samples else 0
        )
        gpu_temp = (
            np.mean(self.sampler.gpu.temp_samples)
            if self.sampler.gpu.temp_samples
            else 0
        )
        gpu_power = (
            np.mean(self.sampler.gpu.power_samples)
            if self.sampler.gpu.power_samples
            else 0
        )

        metrics = {
            "elapsed_time_s": elapsed,
            "num_params": self.num_params,
            "FLOPs_forward": self.flops_forward,
            "FLOPs_backward": flops_backward,
            "FLOPs_total": flops_total,
            "FLOPs_per_sec": flops_per_sec,
            # CPU
            "CPU_usage_%": self.sampler.get_cpu_average() / num_cores,
            "CPU_peak_%": self.sampler.cpu_peak / num_cores,
            "CPU_integral": self.sampler.get_cpu_integral(),
            # RAM
            "RAM_used_MB": (self.sampler.ram_usage - self.baseline_mem) / 1024**2,
            "RAM_integral_MB": self.sampler.get_ram_integral() / 1024**2,
            "RAM_peak_MB": self.sampler.ram_peak / 1024**2,
            "RAM_mean_MB": self.sampler.get_ram_mean() / 1024**2,
            "RAM_median_MB": self.sampler.get_ram_median() / 1024**2,
            "RAM_std_MB": self.sampler.get_ram_std() / 1024**2,
            "RAM_max_MB": self.sampler.get_ram_max() / 1024**2,
            # VMS
            "VMS_used_MB": self.sampler.vms_usage / 1024**2,
            "VMS_peak_MB": self.sampler.vms_peak / 1024**2,
            "VMS_mean_MB": self.sampler.get_vms_mean() / 1024**2,
            "VMS_median_MB": self.sampler.get_vms_median() / 1024**2,
            "VMS_std_MB": self.sampler.get_vms_std() / 1024**2,
            "VMS_max_MB": self.sampler.get_vms_max() / 1024**2,
            # GPU
            "GPU_available": self.sampler.gpu.available,
            "GPU_backend": self.sampler.gpu.backend,
            "GPU_usage_%": gpu_usage,
            "GPU_peak_%": gpu_peak,
            "GPU_peak_MB": gpu_peak_bytes / 1024**2,
            "GPU_mem_used_MB": gpu_mem / 1024**2,
            "GPU_temp_C": gpu_temp,
            "GPU_power_W": gpu_power,
        }

        return metrics


def save_dict_to_csv(data_dict, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)


def run_training_subprocess(batch_size, epochs, precision, model, return_dict):
    """
    Function to run the training subprocess.

    :param batch_size: Batch size for training
    :param epochs: Epochs for training
    :param params: Parameters for the model
    :param return_dict: Dictionary to store results
    """
    training_results, validation_results = run(
        model=model, batch_size=batch_size, epochs=epochs, precision=precision
    )
    return_dict["training"] = training_results
    return_dict["validation"] = validation_results


def get_gp_predictions(X_train, y_values, GRID, task):
    # 1. Normaliza y
    scaler = StandardScaler()
    y_norm = scaler.fit_transform(np.array(y_values).reshape(-1, 1)).flatten()

    # 2. Roda o GP nos dados normalizados
    mu_n, sigma_n, f_min_n = gaussian_process_cbo(X_train, y_norm, GRID, task)

    # 3. Desnormaliza a média para a escala original
    mu_rescaled = scaler.inverse_transform(np.array(mu_n).reshape(-1, 1)).flatten()

    # 4. Ajusta o sigma (o desvio padrão escala linearmente)
    sigma_rescaled = sigma_n * scaler.scale_[0]

    return mu_rescaled, sigma_rescaled, min(y_values)


def cost_fn(model, trial, training_metrics, validation_metrics):

    log2_batch_size, epochs, precision_exp, task = trial
    batch_size = round(2 ** log2_batch_size.item())

    if batch_size < BATCH_SIZE_MIN:
        batch_size = BATCH_SIZE_MIN

    if batch_size > BATCH_SIZE_MAX:
        batch_size = BATCH_SIZE_MAX

    epochs_int = int(epochs)

    ALLOWED_PRECISIONS = np.array([4, 8, 16, 32])

    precision = int(
        ALLOWED_PRECISIONS[np.argmin(np.abs(ALLOWED_PRECISIONS - 2**precision_exp))]
    )

    print(
        "Batch size: ",
        int(batch_size),
        ", Epochs: ",
        epochs_int,
        ", Precision: ",
        precision,
        ", Task: ",
        task,
    )
    # return_dict = {}
    manager = Manager()
    return_dict = manager.dict()
    p = Process(
        target=run_training_subprocess,
        args=(batch_size, epochs_int, precision, model, return_dict),
    )
    p.start()
    p.join()
    # training_results, validation_results = run(
    #     model=model, batch_size=int(batch_size), epochs=int(epochs), precision=int(precision)
    # )
    training_results = return_dict["training"]
    validation_results = return_dict["validation"]

    save_dict_to_csv(training_results, "data/training_results.csv")
    save_dict_to_csv(validation_results, "data/validation_results.csv")

    loss = float(1.0 - validation_results["accuracy"])
    proc = (
        training_results["GPU_usage_%"]
        if DEVICE == "cuda" and torch.cuda.is_available()
        else training_results["CPU_usage_%"]
    )
    mem = (
        training_results["GPU_mem_used_MB"]
        if DEVICE == "cuda" and torch.cuda.is_available()
        else training_results["RAM_used_MB"]
    )
    training_time = training_results["elapsed_time_s"]
    model_size = validation_results["Model_size_MB"]

    proc_penalty = LAMBDA_PROC * (proc / PROC_MAX)
    mem_penalty = LAMBDA_MEM * (mem / MEM_MAX)
    time_penalty = LAMBDA_TIME * (training_time / TRAINING_TIME_MAX)
    loss_penalty = LAMBDA_LOSS * loss
    model_size_penalty = LAMBDA_MODEL_SIZE * (model_size / MODEL_SIZE_LIMIT)

    print(
        f"Proc.: {round(proc,2)}%, Proc Max.: {PROC_MAX}%, Fraction: {round(proc / PROC_MAX,2)}, Proc. Penalty: {round(proc_penalty,2)}"
    )
    print(
        f"Mem. Usage: {round(mem,2)}MB, Mem. Max.: {MEM_MAX}MB, Fraction: {round(mem / MEM_MAX,2)}, Mem. Penalty: {round(mem_penalty,2)}"
    )
    print(
        f"Training Time: {round(training_time,2)}s, Max Training Time: {TRAINING_TIME_MAX}s, Fraction: {round(training_time / TRAINING_TIME_MAX,2)}, Time Penalty: {round(time_penalty,2)}"
    )

    print(
        "Model Size: ",
        round(model_size, 2),
        "MB,",
        f"Model Size Limit: {MODEL_SIZE_LIMIT}MB, Fraction: {round(model_size / MODEL_SIZE_LIMIT,2)}, Model Size Penalty: {round(model_size_penalty,2)}",
    )
    print(f"Loss: {loss:.4e}, Loss Penalty: {round(loss_penalty,2)}")
    print(
        "Total Penalty: ",
        round(
            proc_penalty
            + mem_penalty
            + time_penalty
            + loss_penalty
            + model_size_penalty,
            4,
        ),
        "\n",
    )
    penalty_dict = {
        "total_penalty": proc_penalty
        + mem_penalty
        + time_penalty
        + loss_penalty
        + model_size_penalty,
        "proc": proc,
        "prc_penalty": proc_penalty,
        "mem": mem,
        "mem_penalty": mem_penalty,
        "time": training_time,
        "time_penalty": time_penalty,
        "loss_penalty": loss_penalty,
        "model_size_penalty": model_size_penalty,
        "model_size": model_size,
    }
    training_metrics.append(training_results)
    validation_metrics.append(validation_results)

    return penalty_dict


def sample_initial_points(n_points):
    """
    Samples initial points for Bayesian Optimization.

    :param n_points: Number of initial points to sample
    """
    log2_batch_sizes = np.random.choice(LOG2_BATCH_SIZE_GRID, n_points)
    epochs = np.random.choice(EPOCHS_GRID, n_points)
    log2_precisions = np.random.choice(LOG2_PRECISION_GRID, n_points)
    tasks = np.random.choice(list(TASKS), n_points)

    X = []
    for log_bs, ep, log2_precision, task in zip(
        log2_batch_sizes, epochs, log2_precisions, tasks
    ):
        z = task
        X.append([log_bs, ep, log2_precision, z])

    return X, tasks


def gaussian_process_cbo(X, y, grid_actions, z_t):
    """
    Performs Gaussian Process regression for Contextual Bayesian Optimization.

    :param X: Observed inputs
    :param y: Observed outputs
    :param grid_actions: Grid of actions to evaluate
    :param z_t: Contextual parameter
    """
    n = len(X)

    # K (n x n)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff_x = np.array(X[i][:2]) - np.array(X[j][:2])
            d2_x = diff_x.dot(diff_x)

            d2_task = 0.0 if X[i][2] == X[j][2] else 1.0

            K[i, j] = SIGMA_F2 * math.exp(-0.5 * (d2_x / (ELL**2) + d2_task / (ELL**2)))

    K_noise = K + SIGMA_N2 * np.eye(n)
    K_inv = np.linalg.inv(K_noise)

    f_min = min(y)

    mu_values = []
    sigma_values = []

    for a_cand in grid_actions:
        x_cand = np.array([a_cand[0], a_cand[1], a_cand[2], z_t])

        k_star = np.array(
            [
                SIGMA_F2
                * math.exp(-0.5 * np.sum((np.array(X[i]) - x_cand) ** 2) / (ELL**2))
                for i in range(n)
            ]
        )

        mu = float(k_star.dot(K_inv).dot(np.array(y)))
        # mu = max(mu, 0.0)
        mu_values.append(mu)

        k_ss = SIGMA_F2
        var = k_ss - float(k_star.dot(K_inv).dot(k_star))
        var = max(var, 0.0)
        sigma = math.sqrt(var)
        sigma_values.append(sigma)

    return mu_values, sigma_values, f_min


def expected_improvement(mu: list, sigma: list, f_min: float, grid: list):
    """
    Computes the Expected Improvement (EI) acquisition function.

    :param mu: Mean predictions
    :param sigma: Standard deviation predictions
    :param f_min: Current minimum objective value
    :param grid: Grid of candidate points
    """
    best_ei = -np.inf
    best_x = None
    ei_values = []

    for m, s, x_cand in zip(mu, sigma, grid):
        if s < 1e-12:
            ei = max(0.0, f_min - m)
        else:
            Z = (f_min - m) / s
            ei = (f_min - m) * norm.cdf(Z) + s * norm.pdf(Z)
        ei_values.append(ei)

        if ei > best_ei:
            best_ei = ei
            best_x = x_cand

    return best_x, best_ei, ei_values


def probability_feasible(mu, sigma, limit, eps=1e-12):
    """
    Computes the Probability of Feasibility (PF) for given mean and standard deviation.

    :param mu: Mean predictions
    :param sigma: Standard deviation predictions
    :param limit: Feasibility limit
    :param eps: Small constant to avoid division by zero
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    return norm.cdf((limit - mu) / (sigma + eps))


def init_history(tasks):
    """
    Initializes the history dictionary for tracking optimization results.

    :param tasks: List of tasks
    """
    return {
        t: {
            "penalty": [],
            "proc": [],
            "proc_penalty": [],
            "mem": [],
            "mem_penalty": [],
            "time": [],
            "time_penalty": [],
            "loss": [],
            "model_size": [],
            "model_size_penalty": [],
        }
        for t in tasks
    }


def task_selector(task_name):
    """
    Function to select task parameters based on task name.

    :param task_name: Name of the task
    """
    tasks = {
        1: {
            "hidden_units": [128],
            "dropout_rate": 0.1,
        },
        2: {
            "hidden_units": [512, 256],
            "dropout_rate": 0.2,
        },
        3: {
            "hidden_units": [1024, 1024, 512],
            "dropout_rate": 0.4,
        },
    }
    return tasks[task_name]


def run(
    model: MLP = None,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    precision: int = 32,
):
    """
    Function to run the training of the CNN model.

    :param params: Model Parameters
    :param batch_size: Batch size for training
    :param epochs: Epochs for training
    """
    # loading and preprocessing the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    train_data = datasets.CIFAR10(
        root=".", train=True, download=True, transform=transform
    )

    test_data = datasets.CIFAR10(
        root=".", train=False, download=True, transform=transform
    )

    n_train = 2000
    n_test = 500

    # índices aleatórios (reprodutíveis)
    rng = np.random.default_rng(seed=42)

    train_idx = rng.choice(len(train_data), size=n_train, replace=False)
    test_idx = rng.choice(len(test_data), size=n_test, replace=False)

    train_data_small = Subset(train_data, train_idx)
    test_data_small = Subset(test_data, test_idx)

    train_loader = DataLoader(train_data_small, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_small, batch_size=batch_size, shuffle=False)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and DEVICE == "cuda") else "cpu"
    )

    # Training the MLP model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    acc_metric = MulticlassAccuracy(num_classes=10).to(device)
    acc_metric.reset()
    monitor = Monitor(model=model, device=device)
    monitor.start()
    for _ in tqdm(range(epochs), "Training..."):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc_metric.update(outputs, labels)

    training_metrics = monitor.end()
    training_metrics["accuracy"] = acc_metric.compute().item()

    model2 = copy.deepcopy(model).eval()

    if precision == 32:
        model2 = model2.to(device)
        model_dtype = torch.float32
        eval_device = device

    elif precision == 16:
        model2 = model2.to(device).half()
        model_dtype = torch.float16
        eval_device = device

    elif precision in [8, 4]:

        # method = int8_weight_only() if precision == 8 else int4_weight_only()
        method = int8_weight_only()
        model2 = model2.to(device=device)
        quantize_(model2, method)
        model_dtype = torch.bfloat16
        eval_device = device
    else:
        raise ValueError("Use precision: 4, 8, 16 ou 32")

    test_acc = MulticlassAccuracy(num_classes=10).to(eval_device)
    monitor = Monitor(model=model2, device=eval_device)
    monitor.start()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Eval {precision}-bit"):
            images = images.to(eval_device, dtype=model_dtype)
            labels = labels.to(eval_device)
            outputs = model2(images)
            test_acc.update(outputs, labels)

    prediction_metrics = monitor.end()

    prediction_metrics["accuracy"] = test_acc.compute().item()
    buffer = io.BytesIO()
    torch.save(model2.state_dict(), buffer)
    size_bytes = buffer.getbuffer().nbytes
    model_size_mb = size_bytes / (1024**2)

    prediction_metrics["Model_size_MB"] = model_size_mb
    training_metrics = {
        k: v.item() if isinstance(v, np.generic) else v
        for k, v in training_metrics.items()
    }
    prediction_metrics = {
        k: v.item() if isinstance(v, np.generic) else v
        for k, v in prediction_metrics.items()
    }

    return training_metrics, prediction_metrics


def plot_res_usage(proc, mem, training_time, model_size, t, task, client_id):
    """
    Plots the resource usage radar chart.

    :param proc: Proc. usage percent
    :param mem: Memory usage
    :param training_time: Training time
    :param model_size: Model size in MB
    :param t: Iteration number
    """

    labels = ["Proc. (%)", "Mem. (MB)", "Time (s)", "Model Size (MB)"]
    num_vars = len(labels)
    values = np.array(
        [
            proc / PROC_MAX,
            mem / MEM_MAX,
            training_time / TRAINING_TIME_MAX,
            model_size / MODEL_SIZE_LIMIT,
        ]
    )
    limits = np.ones(num_vars)
    values = np.concatenate([values, [values[0]]])
    limits = np.concatenate([limits, [limits[0]]])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, limits, linewidth=3, linestyle="--", label="Max limits")
    ax.fill(angles, limits, alpha=0.05)
    ax.plot(angles, values, linewidth=2, label="Current usage")
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Resource Usage and Constraints - Iteration {t+1} (Task {task})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"figure/round {t+1} - client {client_id} - radar.png")
    plt.close()


def plot_surrogate(X_train, t, task, EI_mat, MU_mat, client_id):
    """
    Plot surrogate model mean and expected improvement.

    :param X_train: Training data
    :param t: Iteration number
    :param task: Task identifier
    :param EI_mat: Expected Improvement matrix
    :param MU_mat: Mean prediction matrix
    """
    fig = plt.figure(figsize=(14, 12))
    PRECISIONS = [3, 4, 5]

    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    ax2 = fig.add_subplot(2, 1, 2, projection="3d")

    xs = np.array([x for x in X_train if x[3] == task])

    for p_idx, p_val in enumerate(PRECISIONS):
        # --- Camada MU (Média) ---
        cf1 = ax1.contourf(
            LOG2_BATCH_SIZE_GRID,
            EPOCHS_GRID,
            MU_mat[:, :, p_idx].T,
            zdir="z",
            offset=p_val + 0.15,
            levels=30,
            cmap="viridis",
            alpha=0.8,
        )

        # --- Camada EI (Expected Improvement) ---
        cf2 = ax2.contourf(
            LOG2_BATCH_SIZE_GRID,
            EPOCHS_GRID,
            EI_mat[:, :, p_idx].T,
            zdir="z",
            offset=p_val + 0.15,
            levels=30,
            cmap="viridis",
            alpha=0.8,
        )

        # Pontos de treino
        mask = np.abs(xs[:, 2] - p_val) < 0.1
        if np.any(mask):
            ax1.scatter(
                xs[mask, 0],
                xs[mask, 1],
                zs=p_val + 0.15,
                c="r",
                edgecolors="k",
                s=40,
                depthshade=False,
            )
            ax2.scatter(
                xs[mask, 0],
                xs[mask, 1],
                zs=p_val + 0.15,
                c="r",
                edgecolors="k",
                s=40,
                depthshade=False,
            )

    # Colorbars
    cbar1 = fig.colorbar(cf1, ax=ax1, shrink=0.6, aspect=10, pad=0.1)
    cbar1.set_label("Mean Prediction μ(x)")

    cbar2 = fig.colorbar(cf2, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
    cbar2.set_label("Expected Improvement")

    # Configurações do Eixo MU
    ax1.set_title(f"GP Surrogate μ(x) - Iteration {t+1} (Task {task})")
    ax1.set_zlim(min(PRECISIONS) - 1, max(PRECISIONS) + 1)
    ax1.set_zlabel("Log2(Precision)")
    ax1.set_ylabel("Epochs")
    ax1.set_xlabel("log2(Batch Size)")

    # Configurações do Eixo EI
    ax2.set_title("Expected Improvement")
    ax2.set_zlim(min(PRECISIONS) - 1, max(PRECISIONS) + 1)
    ax2.set_zlabel("Log2(Precision)")
    ax2.set_ylabel("Epochs")
    ax2.set_xlabel("log2(Batch Size)")

    # Ajuste de ângulo para melhor visualização
    ax1.view_init(elev=20, azim=-35)
    ax2.view_init(elev=20, azim=-35)

    plt.tight_layout()
    plt.savefig(f"figure/round {t+1} - client {client_id} - surrogate.png")
    plt.close()


def plot_ei(X_train, y_train, t, client_id):
    """
    Plots the 3D stacked surrogate slices for each task and each metric with titles.
    """
    fig = plt.figure(figsize=(22, 24))
    plt.suptitle(
        f"Iteration {t+1} - Multitask 3D Stacked Surrogate Slices", fontsize=24, y=0.98
    )

    n_tasks = len(TASKS)

    for col, target_task in enumerate(TASKS):
        # --- Cálculo dos Modelos (Lógica original convertendo para Array) ---
        mu_obj_t, sigma_obj_t, f_min_t = gaussian_process_cbo(
            X_train, y_train["total_penalty"], GRID, target_task
        )
        mu_proc_t, sigma_proc_t, _ = gaussian_process_cbo(
            X_train, y_train["proc"], GRID, target_task
        )
        mu_mem_t, sigma_mem_t, _ = gaussian_process_cbo(
            X_train, y_train["mem"], GRID, target_task
        )
        mu_time_t, sigma_time_t, _ = gaussian_process_cbo(
            X_train, y_train["time"], GRID, target_task
        )
        mu_size_t, sigma_size_t, _ = gaussian_process_cbo(
            X_train, y_train["model_size"], GRID, target_task
        )

        _, _, ei_scalar_t = expected_improvement(mu_obj_t, sigma_obj_t, f_min_t, GRID)
        p_proc_t = probability_feasible(mu_proc_t, sigma_proc_t, PROC_MAX)
        p_mem_t = probability_feasible(mu_mem_t, sigma_mem_t, MEM_MAX)
        p_time_t = probability_feasible(mu_time_t, sigma_time_t, TRAINING_TIME_MAX)
        p_size_t = probability_feasible(mu_size_t, sigma_size_t, MODEL_SIZE_LIMIT)

        p_feas_t = p_proc_t * p_mem_t * p_time_t * p_size_t
        alpha_t = np.array(ei_scalar_t) * p_feas_t

        # --- Reshape para 3D ---
        shape_3d = (
            len(LOG2_BATCH_SIZE_GRID),
            len(EPOCHS_GRID),
            len(LOG2_PRECISION_GRID),
        )

        matrices = [
            np.array(mu_obj_t).reshape(shape_3d),
            np.array(p_proc_t).reshape(shape_3d),
            np.array(p_mem_t).reshape(shape_3d),
            np.array(p_time_t).reshape(shape_3d),
            np.array(p_size_t).reshape(shape_3d),
            np.array(p_feas_t).reshape(shape_3d),
            np.array(alpha_t).reshape(shape_3d),
        ]

        labels = [
            "μ(x)",
            "p(Proc. <= Proc. Max.)",
            "p(Mem. <= Mem. Max.)",
            "p(Time <= Time Max.)",
            "p(Size <= Size Max.)",
            "Joint Feasibility",
            "EI(x)",
        ]

        xs_t = np.array([x for x in X_train if x[2] == target_task])

        for row in range(len(matrices)):
            idx = row * n_tasks + col + 1
            ax = fig.add_subplot(len(matrices), n_tasks, idx, projection="3d")
            current_mat = matrices[row]

            for p_idx, p_val in enumerate(LOG2_PRECISION_GRID):

                z_data = current_mat[:, :, p_idx].T
                if row > 0 and row < 6:
                    if np.max(z_data) - np.min(z_data) < 1e-7:
                        levels = np.linspace(0, 1, 31)
                    else:
                        levels = 30
                else:
                    levels = 30
                cf = ax.contourf(
                    LOG2_BATCH_SIZE_GRID,
                    EPOCHS_GRID,
                    current_mat[:, :, p_idx].T,
                    zdir="z",
                    offset=p_val + 0.15,
                    levels=levels,
                    cmap="viridis",
                    alpha=0.8,
                    vmin=0 if 0 < row < 6 else None,
                    vmax=1 if 0 < row < 6 else None,
                )

                # Pontos de treino nas linhas de Média e Aquisição
                if row == 0 or row == 6:
                    if xs_t.size > 0:
                        mask = np.abs(xs_t[:, 3] - p_val) < 0.1
                        if np.any(mask):
                            ax.scatter(
                                xs_t[mask, 0],
                                xs_t[mask, 1],
                                zs=p_val + 0.15,
                                c="r",
                                edgecolors="k",
                                s=25,
                                depthshade=False,
                            )

                # Linha de viabilidade 0.5 na Aquisição
                if row == 6:
                    ax.contour(
                        LOG2_BATCH_SIZE_GRID,
                        EPOCHS_GRID,
                        matrices[5][:, :, p_idx].T,
                        levels=[0.5],
                        colors="red",
                        linewidths=1.5,
                        zdir="z",
                        offset=p_val + 0.15,
                    )

            # --- Títulos e Estética ---
            # Título superior indica a Task
            if row == 0:
                ax.set_title(
                    f"TASK: {target_task}\n{labels[row]}",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                ax.set_title(
                    labels[row],
                    fontsize=14,
                    fontweight="bold",
                )

            # Labels de eixos apenas nas bordas para não poluir
            # if col == 0:
            #     ax.set_zlabel("Precision", fontsize=10)
            #     ax.set_ylabel("Epochs", fontsize=10)
            # if row == 6:
            #     ax.set_xlabel("log2(Batch size)", fontsize=10)
            ax.set_ylabel("Epochs", fontsize=10)
            ax.set_xlabel("log2(Batch size)", fontsize=10)
            ax.set_zlabel("Precision", fontsize=10)

            ax.set_zlim(min(LOG2_PRECISION_GRID) - 1, max(LOG2_PRECISION_GRID) + 1)
            ax.set_zticks(LOG2_PRECISION_GRID)
            ax.view_init(elev=20, azim=-35)

            # Barra de cores
            cbar = fig.colorbar(cf, ax=ax, shrink=0.5, aspect=12, pad=0.1)
            cbar.set_label(labels[row])
            cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"figure/round {t+1} - client {client_id} - constraints.png", dpi=150)
    plt.close()


def final_plot(history, client_id):
    """
    Plots the final history of loss, resource usage, and penalty.

    :param history: History dictionary containing optimization results
    """
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for i, task in enumerate(TASKS):
        ax = axes[i]
        history_loss = history[task]["loss"][4:]
        ax.plot(
            history_loss,
            marker="o",
            color="tab:red",
            linestyle="-",
            label=f"Loss Task {task}",
        )
        ax.set_title(f"Loss Penalty - Task {task}", fontsize=12)
        ax.set_xlabel("Iteration")
        if i == 0:
            ax.set_ylabel("Loss Penalty Value")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"figure/client {client_id} - loss_history.png", dpi=300)
    plt.close()

    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5), sharey=True)

    if n_tasks == 1:
        axes = [axes]

    for i, task in enumerate(TASKS):
        ax = axes[i]

        history_proc = [v / PROC_MAX for v in history[task]["proc"]]
        history_mem = [v / MEM_MAX for v in history[task]["mem"]]
        history_time = [v / TRAINING_TIME_MAX for v in history[task]["time"]]
        history_model_size = [v / MODEL_SIZE_LIMIT for v in history[task]["model_size"]]

        ax.plot(history_proc[4:], label="Proc.", marker="o", linestyle="-")
        ax.plot(history_mem[4:], label="Mem.", marker="o", linestyle="-")
        ax.plot(history_time[4:], label="Time", marker="o", linestyle="-")
        ax.plot(history_model_size[4:], label="Model Size", marker="o", linestyle="-")
        ax.axhline(y=1.0, linestyle="--", linewidth=2, color="red", alpha=0.7)
        ax.set_title(f"Resource Usage - Task {task}", fontsize=12)
        ax.set_xlabel("Iteration")
        if i == 0:
            ax.set_ylabel("Normalized Usage (Value / Limit)")

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"figure/client {client_id} - resources_history.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for i, task in enumerate(TASKS):
        ax = axes[i]
        history_penalty = history[task]["penalty"][4:]
        ax.plot(
            history_penalty,
            marker="o",
            color="darkmagenta",
            linestyle="-",
            label=f"Penalty Task {task}",
        )
        ax.set_title(f"Constraint Penalty - Task {task}", fontsize=12)
        ax.set_xlabel("Iteration")

        if i == 0:
            ax.set_ylabel("Penalty Value")

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"figure/client {client_id} - penalty_history.png", dpi=300)
    plt.close()


def compute_step(X_train, y_train, task):
    """
    Computes the next step in Bayesian optimization.

    :param X_train: Training data points
    :param y_train: Training data values
    :param task: Current task being optimized
    """
    mu_obj, sigma_obg, f_min = gaussian_process_cbo(
        X_train, y_train["total_penalty"], GRID, task
    )
    mu_proc, sigma_proc, _ = gaussian_process_cbo(X_train, y_train["proc"], GRID, task)
    mu_mem, sigma_mem, _ = gaussian_process_cbo(X_train, y_train["mem"], GRID, task)
    mu_time, sigma_time, _ = gaussian_process_cbo(X_train, y_train["time"], GRID, task)
    mu_size, sigma_size, _ = gaussian_process_cbo(
        X_train, y_train["model_size"], GRID, task
    )

    _, _, ei_scalar = expected_improvement(mu_obj, sigma_obg, f_min, GRID)
    p_proc = probability_feasible(mu_proc, sigma_proc, PROC_MAX)
    p_mem = probability_feasible(mu_mem, sigma_mem, MEM_MAX)
    p_time = probability_feasible(mu_time, sigma_time, TRAINING_TIME_MAX)
    p_size = probability_feasible(mu_size, sigma_size, MODEL_SIZE_LIMIT)
    p_feasible = p_proc * p_mem * p_time * p_size
    alpha = ei_scalar * p_feasible

    best_idx = np.argmax(alpha)
    best_x = GRID[best_idx]
    x_new = [best_x[0], best_x[1], best_x[2], task]

    return x_new, mu_obj, alpha


# %%
# APRENDIZADO FEDERADO
def run_test(model: nn.Module, precision: int):
    """
    Function to run the evaluation of the CNN model.

    :param params: Model Parameters
    :param precision: Precision
    """
    # loading and preprocessing the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    test_data = datasets.CIFAR10(
        root=".", train=False, download=False, transform=transform
    )
    n_test = 100
    rng = np.random.default_rng()

    test_idx = rng.choice(len(test_data), size=n_test, replace=False)
    test_data_small = Subset(test_data, test_idx)
    test_loader = DataLoader(test_data_small, batch_size=128, shuffle=False)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and DEVICE == "cuda") else "cpu"
    )
    model2 = copy.deepcopy(model).eval()

    if precision == 32:
        model2 = model2.to(device)
        model_dtype = torch.float32
        eval_device = device

    elif precision == 16:
        model2 = model2.to(device).half()
        model_dtype = torch.float16
        eval_device = device

    elif precision in [8, 4]:
        # method = int8_weight_only() if precision == 8 else int4_weight_only()
        method = int8_weight_only()
        model2 = model2.to(device=device)
        quantize_(model2, method)
        model_dtype = torch.bfloat16
        eval_device = device
    else:
        raise ValueError("Use precision: 4, 8, 16 ou 32")

    test_acc = MulticlassAccuracy(num_classes=10).to(eval_device)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Eval {precision}-bit"):
            images = images.to(eval_device, dtype=model_dtype)
            labels = labels.to(eval_device)
            outputs = model2(images)
            test_acc.update(outputs, labels)

    test_metrics = {
        "loss": 1.0 - test_acc.compute().item(),
        "records": len(test_data_small),
        "accuracy": test_acc.compute().item(),
    }
    return test_metrics


def run_test_subprocess(model, precision, return_dict):
    test_results = run_test(model=model, precision=precision)
    return_dict["test"] = test_results


def test_fn(model: nn.Module, precision: int):
    manager = Manager()
    return_dict = manager.dict()
    p = Process(
        target=run_test_subprocess,
        args=(model, precision, return_dict),
    )
    p.start()
    p.join()
    # return_dict = {}
    # test_results = run_test(model=model, precision=precision)
    # return_dict["test"] = test_results
    test_results = return_dict["test"]
    return test_results
