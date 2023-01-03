import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class Collector:
    def __init__(self, model):
        self.metrics = {
            "rewards": [],
            "ndcg_next_item": [],
            "ndcg_all_item": [],
        }

        if model == "ddpg":
            self.loss = {
                "actor_loss": [],
                "critic_loss": []
            }
        elif model == "bcq":
            self.loss = {
                "generator_loss": [],
                "perturbator_loss": [],
                "critic_loss": [],
                "y": [],
                "q1": [],
                "q2": [],
                "mean": [],
                "std": []
            }
        elif model == "reinforce":
            self.loss = {
                "policy_loss": [],
                "beta_loss": [],
                "importance_weight": [],
                "lambda_k": []
            }

    def gather_info(self, info):
        for k, v in info.items():
            if k in self.metrics:
                self.metrics[k].append(v)
            elif k in self.loss:
                if v is not None:
                    self.loss[k].append(v)

    def print_and_clear_info(self, train_flag = True, epoch = None):
        print()
        for k, v in self.loss.items():
            print(f"{k}: {np.mean(v):.4f}", end=", ")

        for k, v in self.metrics.items():
            if k == "rewards":
                print(f"\nreward: {sum(self.metrics['rewards'])}", end=", ")
                if train_flag:
                    writer.add_scalar('rewards/train', sum(self.metrics['rewards']),epoch)
                else:
                    writer.add_scalar('rewards/eval', sum(self.metrics['rewards']), epoch)
            elif k in ("ndcg_next_item", "ndcg_all_item"):
                print(f"{k}: {np.mean(v):.6f}", end=", ")
                if k == "ndcg_next_item":
                    if train_flag:
                        writer.add_scalar('ndcg_next_item/train', np.mean(v), epoch)
                    else:
                        writer.add_scalar('ndcg_next_item/eval', np.mean(v), epoch)
                else:
                    if train_flag:
                        writer.add_scalar('ndcg_all_item/train', np.mean(v), epoch)
                    else:
                        writer.add_scalar('ndcg_all_item/eval', np.mean(v), epoch)

        print(f"ndcg: {self.metrics['ndcg']:.6f}")
        if train_flag:
            writer.add_scalar('ndcg/train', self.metrics['ndcg'], epoch)
        else:
            writer.add_scalar('ndcg/eval', self.metrics['ndcg'], epoch)
        for k in self.metrics:
            self.metrics[k] = []
        for k in self.loss:
            self.loss[k] = []
