import sys
from . import classification

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m my_package <command> [hydra_args...]")
        print("\nAvailable commands:")
        print("  train-classifier  train a classification model")
        sys.exit(1)

    command = sys.argv[1]

    # CRITICAL: Remove the subcommand from sys.argv so Hydra doesn't trip over it.
    # e.g., ["__main__.py", "script-a", "lr=0.01"] -> ["__main__.py", "lr=0.01"]
    sys.argv.pop(1)

    # Route to the unmodified Hydra scripts
    if command == "train-classifier":
        classification.train.main()
    elif command == "gen-model-configs":
        assert len(sys.argv) == 2, "Please provide a config directory as an argument."
        from .conf import gen_octic_config, gen_honey_config
        gen_octic_config.generate(config_dir=sys.argv[1])
        gen_honey_config.generate(config_dir=sys.argv[1])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()