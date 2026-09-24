import flwr as fl
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar
import os

history_per_task = {
    1: {"rounds": [], "loss": []},
    2: {"rounds": [], "loss": []},
    3: {"rounds": [], "loss": []},
}


def update_plot():
    plt.figure(figsize=(10, 6))

    colors = {1: "blue", 2: "green", 3: "red"}
    markers = {1: "o", 2: "s", 3: "D"}

    for task_id, data in history_per_task.items():
        if data["rounds"]:
            plt.plot(
                data["rounds"],
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


class MultiTaskStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Scalar], Dict[str, Scalar]]:

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if loss is not None:
            current_task = ((server_round - 1) % 3) + 1

            history_per_task[current_task]["rounds"].append(server_round)
            history_per_task[current_task]["loss"].append(loss)

            update_plot()
            print(
                f"-> Round {server_round}: Gráfico atualizado para Task {current_task}"
            )

        return loss, metrics


def select_task(round_number: int) -> dict:
    return {"task": ((round_number - 1) % 3) + 1}


if __name__ == "__main__":
    if os.path.exists("figure/federated_learning_loss.png"):
        os.remove("figure/federated_learning_loss.png")

    strategy = MultiTaskStrategy(
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        on_fit_config_fn=select_task,
        on_evaluate_config_fn=select_task,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=50),
    )
