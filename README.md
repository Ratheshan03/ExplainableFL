# ExplainableFL

ExplainableFL is a Python package designed to bring explainability to Federated Learning models using SHAP values. It provides easy-to-use methods to visualize the impact of model features and privacy mechanisms on model performance.

## Installation

To install ExplainableFL, run the following command:

```bash
pip install -i https://test.pypi.org/simple/ explainablefl
```

Ensure you have the necessary prerequisites installed, including Python 3.6+ and pip

## Usage

Here's a quick example to get you started:

```import torch
from torch.utils.data import DataLoader, TensorDataset
from explainablefl import FederatedXAI

# Example model
model = torch.nn.Linear(10, 2)

# Setup DataLoader
x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=10)

# Initialize FederatedXAI
federated_xai = FederatedXAI(device=torch.device('cpu'), global_model=model, data_loader=data_loader)

# Use the library to explain the client model
shap_plot_buf, _ = federated_xai.explain_client_model(model)
```

## Features

- Client Model Explanation: Visualize how individual features influence model predictions.
- Global Model Explanation: Generate SHAP explanations and confusion matrices.
- Aggregation Impact: Assess the effect of model aggregation in federated settings.
- Privacy Impact Analysis: Understand the influence of differential privacy mechanisms.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the documentation, code quality, or add new features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
