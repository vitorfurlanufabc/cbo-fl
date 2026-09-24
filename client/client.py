import torch
import argparse
import flwr as fl
from collections import OrderedDict
from utils import *
import torch.multiprocessing as mp


class FederatedClient(fl.client.NumPyClient):
    """Federated learning client using CBO to tune local training hyperparameters."""

    def __init__(self, partition_id: int):
        """
        Initialize federated client with models for each task and BO state.

        :param partition_id: Unique integer identifier for this client partition.
        """
        super().__init__()
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and DEVICE == "cuda") else "cpu"
        )
        self.model_task1 = MLP(
            hidden_units=[128],
            dropout_rate=0.1,
        ).to(self.device)
        self.model_task2 = MLP(
            hidden_units=[512, 512],
            dropout_rate=0.2,
        ).to(self.device)
        self.model_task3 = MLP(
            hidden_units=[1024, 1024, 512],
            dropout_rate=0.4,
        ).to(self.device)
        self.current_model = self.model_task1
        self.partition_id = partition_id
        self.epochs = None
        self.batch_size = None
        self.precision = 32
        self.bo_initialized = False
        self.X_train = []
        self.y_train = {...}
        self.history = init_history(TASKS)
        self.current_x = None
        self.current_task = None
        self.training_metrics = []
        self.validation_metrics = []
        self.round = 0

    def task_to_model(self, task: int):
        """
        Set current_model to the model associated with the given task.

        :param task: Integer task identifier (1, 2, or 3).
        """
        if task == 1:
            self.current_model = self.model_task1
        elif task == 2:
            self.current_model = self.model_task2
        elif task == 3:
            self.current_model = self.model_task3
        else:
            raise ValueError("Invalid task number")

    def get_parameters(self, config=None):
        precision = config["precision"] if config else 32
        model2 = copy.deepcopy(self.current_model).eval()

        if precision == 32:
            model2 = model2.to(self.device)

        elif precision == 16:
            model2 = model2.to(self.device).half()

        elif precision in [8, 4]:
            # method = int8_weight_only() if precision == 8 else int4_weight_only()
            method = int8_weight_only()
            model2 = model2.to(device=self.device)
            quantize_(model2, method)
        else:
            raise ValueError("Use precision: 4, 8, 16 ou 32")

        try:
            return [
                (
                    val.dequantize().detach().cpu().numpy()
                    if hasattr(val, "dequantize")
                    else val.detach().cpu().numpy()
                )
                for _, val in model2.state_dict().items()
            ]
        except Exception as e:
            print("Error on get model params:")
            print(e)

    def set_model_params(self, params):
        total_params_received = sum(v.size for v in params)

        models_available = {
            1: self.model_task1,
            2: self.model_task2,
            3: self.model_task3,
        }

        selected_model = None

        for _, model in models_available.items():
            model_size = sum(p.numel() for p in model.state_dict().values())
            if model_size == total_params_received:
                selected_model = model
                break

        if selected_model is None:
            selected_model = models_available.get(self.current_task)

        try:
            keys = selected_model.state_dict().keys()
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
            selected_model.load_state_dict(state_dict, strict=True)
            self.current_model = selected_model

        except Exception as e:
            print(f"Erro crítico ao carregar parâmetros: {e}")

    def fit(self, parameters, config):
        self.current_task = config["task"]
        self.task_to_model(self.current_task)

        self.set_model_params(parameters)

        if not self.bo_initialized:
            log2_batch_size = np.random.choice(LOG2_BATCH_SIZE_GRID)
            epochs = np.random.choice(EPOCHS_GRID)
            precisions = np.random.choice(LOG2_PRECISION_GRID)

            self.X_train = [[log2_batch_size, epochs, precisions, self.current_task]]

            penalties = cost_fn(
                self.current_model,
                self.X_train[0],
                self.training_metrics,
                self.validation_metrics,
            )
            self.y_train = {k: [v] for k, v in penalties.items()}

            self.current_x = self.X_train[0]

            self.bo_initialized = True
        else:
            x, mu_obj, alpha = compute_step(
                self.X_train, self.y_train, self.current_task
            )
            self.current_x = x
            penalties = cost_fn(
                self.current_model,
                self.current_x,
                training_metrics=self.training_metrics,
                validation_metrics=self.validation_metrics,
            )
            self.X_train.append(self.current_x)
            for k, v in penalties.items():
                self.y_train[k].append(v)

                if k in self.history[self.current_task]:
                    self.history[self.current_task][k].append(v)

            EI_mat = np.array(alpha).reshape(
                len(LOG2_BATCH_SIZE_GRID), len(EPOCHS_GRID), len(LOG2_PRECISION_GRID)
            )
            MU_mat = np.array(mu_obj).reshape(
                len(LOG2_BATCH_SIZE_GRID), len(EPOCHS_GRID), len(LOG2_PRECISION_GRID)
            )
            proc = penalties["proc"]
            mem = penalties["mem"]
            training_time = penalties["time"]
            model_size = penalties["model_size"]
            print("Plotting resource usage, surrogate model and EI...")
            plot_res_usage(
                proc,
                mem,
                training_time,
                model_size,
                self.round,
                self.current_task,
                self.partition_id,
            )
            plot_surrogate(
                self.X_train,
                self.round,
                self.current_task,
                EI_mat,
                MU_mat,
                self.partition_id,
            )
            plot_ei(self.X_train, self.y_train, self.round, self.partition_id)
            self.round += 1

        self.batch_size = int(2 ** self.current_x[0])
        self.epochs = int(self.current_x[1])
        self.precision = int(2 ** self.current_x[2])
        train_loss = penalties["total_penalty"]

        self.history[self.current_task]["penalty"].append(penalties["total_penalty"])
        self.history[self.current_task]["loss"].append(penalties["loss_penalty"])

        return (
            self.get_parameters({"precision": self.precision}),
            2000,
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        self.current_task = config["task"]
        self.task_to_model(self.current_task)
        self.set_model_params(parameters)
        self.current_model.to(self.device)

        test_results = test_fn(self.current_model, self.precision)
        return (
            test_results["loss"],
            test_results["records"],
            {"accuracy": test_results["accuracy"]},
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    N_CLIENTS = 4

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id
    client = FederatedClient(partition_id=partition_id)
    fl.client.start_client(
        server_address="vf54_fl_server:8080",
        client=client.to_client(),
    )

    final_plot(client.history, partition_id)
