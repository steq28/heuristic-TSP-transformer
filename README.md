# TSP Transformer Model for Solving the Traveling Salesman Problem

This repository contains a Jupyter Notebook (`notebook.ipynb`) that implements a Transformer-based model to solve the Traveling Salesman Problem (TSP). The TSP is a classic optimization problem where the goal is to find the shortest possible route that visits a set of cities exactly once and returns to the origin city.

## Overview

The notebook is structured as follows:

1. **Introduction**: A brief introduction to the Traveling Salesman Problem and the motivation behind using a Transformer model to solve it.

2. **Packages**: Lists the necessary Python packages required to run the notebook, including `networkx`, `torch`, and `matplotlib`.

3. **Helper Functions**: Contains utility functions for computing tour lengths, running greedy algorithms, generating random tours, and evaluating the Transformer model on TSP instances.

4. **Dataset & DataLoader**: Defines the `TSPDataset` class for loading and processing TSP data, and sets up DataLoaders for training, validation, and testing.

5. **Model**: Implements the `TSPTransformer` model, which is a Transformer-based neural network designed to solve TSP instances. The model includes positional encoding and a custom architecture tailored for TSP.

6. **Training**:

    - **Without Gradient Accumulation**: Trains the Transformer model using a standard training loop.
    - **With Gradient Accumulation**: Trains the model using gradient accumulation, which is useful for handling large batch sizes with limited memory.

7. **Evaluation**: Evaluates the trained model on the test set and reports the final test loss.

8. **Visualization**: Plots the training and validation losses to monitor the model's performance over epochs.

## Requirements

To run this notebook, you need the following Python packages:

-   `networkx`
-   `torch`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `pickle`

You can install these packages using `pip`:

```bash
pip install networkx torch numpy matplotlib seaborn
```

## Usage

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/steq28/heuristic-TSP-transformer
    cd heuristic-TSP-transformer
    ```

2. **Open the Notebook**:

    ```bash
    jupyter notebook notebook.ipynb
    ```

3. **Run the Notebook**:
    - Execute each cell in the notebook sequentially to load the data, define the model, train it, and evaluate its performance.
    - The notebook includes both standard training and training with gradient accumulation.

## Results

The notebook provides visualizations of the training and validation losses, which help in understanding the model's learning process. The final evaluation on the test set gives an indication of the model's performance in solving TSP instances.

## Acknowledgments

-   The Transformer model architecture is inspired by the original Transformer paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.
-   The TSP dataset used in this notebook is a synthetic dataset generated for demonstration purposes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
