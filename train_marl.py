"""Train and save the centralized-training/decentralized-execution policy."""

from pathlib import Path

from marl import train_ctde


if __name__ == "__main__":
    destination = Path(__file__).with_name("marl_policy.pt")
    result = train_ctde(destination)
    print(f"saved: {destination}")
    print(f"final batch role accuracy: {result['final_batch_role_accuracy']:.3f}")
