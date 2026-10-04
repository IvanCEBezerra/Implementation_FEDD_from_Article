from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def main():
    # Input and output dirs
    base_dir = Path("data/generated/linear")
    output_dir = Path("data/generated/plots/linear")
    
    # Create output folders
    (output_dir / "abrupt").mkdir(parents=True, exist_ok=True)
    (output_dir / "gradual").mkdir(parents=True, exist_ok=True)

    print("Gerando gráficos e salvando como imagens...")

    # Find all .npy files
    for npy_file in base_dir.rglob("*.npy"):
        # Load series
        series = np.load(npy_file)
        
        # Drift type from parent folder
        drift_type = npy_file.parent.name
        
        plt.figure(figsize=(10, 4))
        plt.plot(series, color='black', linewidth=0.8)
        
        # Drifts every 3000 points
        drifts = [3000, 6000, 9000]
        for drift in drifts:
            plt.axvline(x=drift, color='red', linestyle='-', linewidth=1.5, alpha=0.7)

        plt.title(f"Linear Time Series - {npy_file.stem} ({drift_type.capitalize()} Drifts)")
        plt.xlabel("Tempo (Instâncias)")
        plt.ylabel("Valor (xt)")
        plt.xlim(0, 12000)
        plt.tight_layout()
        
        # Save image
        save_path = output_dir / drift_type / f"{npy_file.stem}.png"
        plt.savefig(save_path, dpi=150) # Good resolution
        
        # Close figure to free memory
        plt.close()

    print(f"Pronto! Todos os gráficos foram salvos em: {output_dir}")

if __name__ == "__main__":
    main()