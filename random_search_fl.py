# %%
# imports
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
from concurrent.futures import ThreadPoolExecutor
import threading

plot_lock = threading.Lock()
from utils import *
import torch, time
from collections import OrderedDict


# %%
# classes and functions
class Client:
    def __init__(self, id):
        self.partition_id = id
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and DEVICE == "cuda") else "cpu"
        )
        self.local_model_dict = {
            1: {
                "model": MLP(
                    hidden_units=[128],
                    dropout_rate=0.1,
                ).to(self.device),
                "loss": None,
            },
            2: {
                "model": MLP(
                    hidden_units=[512, 1024],
                    dropout_rate=0.2,
                ).to(self.device),
                "loss": None,
            },
            3: {
                "model": MLP(
                    hidden_units=[512, 1024, 1024],
                    dropout_rate=0.4,
                ).to(self.device),
                "loss": None,
            },
        }
        self.partition_id = id
        self.current_model = None
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
        self.current_round = -1

    def get_parameters(self, config=None):
        precision = config["precision"] if config else 32
        model2 = copy.deepcopy(self.current_model).eval()

        if precision == 32:
            model2 = model2.to(self.device)

        elif precision == 16:
            model2 = model2.to(self.device).half()

        elif precision == 8:
            method = int8_weight_only()
            model2 = model2.to(device=self.device)
            quantize_(model2, method)
        else:
            raise ValueError("Use precision: 8, 16, 32")

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

    def set_model_params(self, params, config):
        task = config["task"]
        keys = self.local_model_dict[task]["model"].state_dict().keys()
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
        self.local_model_dict[task]["model"].load_state_dict(state_dict, strict=True)

    def task_to_model(self, task: int):
        self.current_model = self.local_model_dict[task]["model"]

    def fit(self, parameters, config):
        self.current_task = config["task"]
        self.task_to_model(self.current_task)
        self.set_model_params(parameters, config)

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

        self.batch_size = int(2 ** self.current_x[0])
        self.epochs = int(self.current_x[1])
        self.precision = int(2 ** self.current_x[2])
        self.history[self.current_task]["penalty"].append(penalties["total_penalty"])
        self.history[self.current_task]["loss"].append(penalties["loss_penalty"])
        self.history[self.current_task]["proc"].append(penalties["proc"])
        self.history[self.current_task]["mem"].append(penalties["mem"])
        self.history[self.current_task]["time"].append(penalties["time"])
        self.history[self.current_task]["model_size"].append(penalties["model_size"])

        train_loss = penalties["total_penalty"]

        return (
            self.get_parameters({"precision": self.precision}),
            2000,
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):

        self.current_task = config["task"]
        self.task_to_model(self.current_task)
        self.set_model_params(parameters, config)

        test_results = test_fn(self.current_model, self.precision)

        return (
            test_results["loss"],
            test_results["records"],
            {"mae": test_results["mae"]},
        )

    def update_local_model(self, server, parameters, config):
        print(f"Modelo global recebido pelo cliente {self.partition_id}")
        self.current_task = config["task"]
        parameters, training_records, training_loss = self.fit(parameters, config)

        eval_loss, eval_records, eval_metrics = self.evaluate(parameters, config)
        metrics = {
            "parameters": parameters,
            "loss": eval_loss,
            "training_records": training_records,
            "eval_records": eval_records,
        }
        server.process_client_reply(self, metrics, config["task"])
        self.current_round += 1


class Server:
    def __init__(self):
        self.min_clients = 4
        self.current_round = 0
        self.current_task = 1
        self.global_model_dict = {
            1: {
                "model": MLP(
                    hidden_units=[128],
                    dropout_rate=0.1,
                ),
                "loss": None,
            },
            2: {
                "model": MLP(
                    hidden_units=[512, 1024],
                    dropout_rate=0.2,
                ),
                "loss": None,
            },
            3: {
                "model": MLP(
                    hidden_units=[512, 1024, 1024],
                    dropout_rate=0.4,
                ),
                "loss": None,
            },
        }
        self.connected_clients = []
        self.local_models_dict = {
            1: {"clients": [], "metrics": []},
            2: {"clients": [], "metrics": []},
            3: {"clients": [], "metrics": []},
        }
        self.history_dict = {
            1: {"round": [], "loss": []},
            2: {"round": [], "loss": []},
            3: {"round": [], "loss": []},
        }

    def register_client(self, client):
        self.connected_clients.append(client)

    def process_client_reply(self, client, metrics, task):
        print(f"Modelo local do cliente {client.partition_id} recebido")
        self.local_models_dict[task]["clients"].append(client)
        self.local_models_dict[task]["metrics"].append(metrics)

    def waiting_local_models(self):
        return (
            len(self.local_models_dict[self.current_task]["clients"]) < self.min_clients
        )

    def aggregate_models(self):
        print(f"Executando agregação FedAvg para Task {self.current_task}")
        clients_metrics = self.local_models_dict[self.current_task]["metrics"]

        if not clients_metrics:
            return

        local_models = [m["parameters"] for m in clients_metrics]
        local_weights = [m["training_records"] for m in clients_metrics]
        total_records = sum(local_weights)

        global_model_obj = self.global_model_dict[self.current_task]["model"]
        global_keys = list(global_model_obj.state_dict().keys())

        avg_state_dict = {
            key: torch.zeros_like(val)
            for key, val in global_model_obj.state_dict().items()
        }

        for i, client_params in enumerate(local_models):
            weight_factor = local_weights[i] / total_records

            for key_idx, key_name in enumerate(global_keys):
                client_layer_tensor = torch.tensor(client_params[key_idx])
                avg_state_dict[key_name] += client_layer_tensor * weight_factor

        global_model_obj.load_state_dict(avg_state_dict)

        global_loss = 0
        total_records = 0

        for m in clients_metrics:
            global_loss += m["loss"] * m["eval_records"]
            total_records += m["eval_records"]

        global_loss /= total_records

        self.history_dict[self.current_task]["round"].append(self.current_round)
        self.history_dict[self.current_task]["loss"].append(global_loss)

        print(f"Agregação concluída. Loss Global: {global_loss:.4f}")

        self.update_plot()
        self.local_models_dict[self.current_task] = {"clients": [], "metrics": []}
        self.current_round += 1
        self.current_task = ((self.current_round) % 3) + 1

    def broadcast_global_model(self):
        print(
            f"\n================ Round {self.current_round}: Distribuindo modelo global ================"
        )
        config = {"task": self.current_task}
        model = self.global_model_dict[self.current_task]["model"]
        parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        with ThreadPoolExecutor(max_workers=len(self.connected_clients)) as executor:
            for client in self.connected_clients:
                executor.submit(client.update_local_model, self, parameters, config)

    def update_plot(self):
        plt.figure(figsize=(10, 6))

        colors = {1: "blue", 2: "green", 3: "red"}
        markers = {1: "o", 2: "s", 3: "D"}

        for task_id, data in self.history_dict.items():
            if data["round"]:
                plt.plot(
                    data["round"],
                    data["loss"],
                    label=f"Task {task_id}",
                    color=colors[task_id],
                    marker=markers[task_id],
                    linestyle="-",
                )

        plt.title("Global loss per round")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig("figure/federated_learning_loss.png")
        plt.close()


# %%
# main
if __name__ == "__main__":
    server = Server()
    c0 = Client(0)
    c1 = Client(1)
    c2 = Client(2)
    c3 = Client(3)
    server.register_client(c0)
    server.register_client(c1)
    server.register_client(c2)
    server.register_client(c3)

    while server.current_round < ROUNDS:
        server.broadcast_global_model()
        while server.waiting_local_models():
            time.sleep(1)
        server.aggregate_models()

# %%
