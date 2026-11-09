"""
Example usage of PruneQuest via the Python API.
"""

from prunequest import PruneQuest

def main():
    print("Initializing PruneQuest...")
    quest = PruneQuest(
        model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        benchmark="gsm8k",
        retain=0.99,            # Require at least 99% baseline accuracy retention
        max_iterations=3,       # Maximum layer removals
        output_dir="runs/api_example_run",
    )

    print("Running automated pruning search...")
    result = quest.run()

    print("\n--- Pruning Search Complete ---")
    print(f"Status: {result.status}")
    print(f"Original Parameters: {result.original_params:,}")
    print(f"Final Parameters:    {result.final_params:,}")
    print(f"Baseline Accuracy:   {result.baseline_accuracy*100:.2f}%")
    print(f"Final Accuracy:      {result.final_accuracy*100:.2f}%")
    print(f"Accuracy Retained:   {result.accuracy_retention_pct:.2f}%")
    print(f"Best Checkpoint:     {result.best_checkpoint}")
    print(f"Report Generated:    {result.report_path}")

if __name__ == "__main__":
    main()
