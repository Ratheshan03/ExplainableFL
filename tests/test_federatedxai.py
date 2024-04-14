import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
from explainablefl.explainablefl import FederatedXAI  # Adjust import according to your package structure

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(784, 10)  # Simple linear model for MNIST

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class TestFederatedXAI(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup global model
        self.global_model = SimpleModel().to(self.device)
        
        # Setup MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.data_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)

        # Initialize FederatedXAI with a dummy DataLoader
        self.federated_xai = FederatedXAI(self.device, self.global_model, self.data_loader)

        # Create a directory for test outputs if it doesn't exist
        self.output_dir = "./test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def test_explain_client_model(self):
        client_model = SimpleModel().to(self.device)
        shap_plot_buf, _ = self.federated_xai.explain_client_model(client_model)
        plot_path = os.path.join(self.output_dir, "client_model_shap.png")
        with open(plot_path, "wb") as f:
            f.write(shap_plot_buf.getbuffer())
        self.assertTrue(os.path.exists(plot_path), "SHAP plot should be saved to a file")

    def test_explain_global_model(self):
        buf_conf_matrix, shap_plot_buf, _ = self.federated_xai.explain_global_model()
        conf_matrix_path = os.path.join(self.output_dir, "global_model_conf_matrix.png")
        shap_plot_path = os.path.join(self.output_dir, "global_model_shap.png")
        with open(conf_matrix_path, "wb") as f:
            f.write(buf_conf_matrix.getbuffer())
        with open(shap_plot_path, "wb") as f:
            f.write(shap_plot_buf.getbuffer())
        self.assertTrue(os.path.exists(conf_matrix_path), "Confusion matrix should be saved to a file")
        self.assertTrue(os.path.exists(shap_plot_path), "SHAP plot should be saved to a file")

    def test_explain_aggregation(self):
        plot_buffer = self.federated_xai.explain_aggregation(self.global_model.state_dict(), self.global_model.state_dict())
        aggregation_plot_path = os.path.join(self.output_dir, "aggregation_shap.png")
        with open(aggregation_plot_path, "wb") as f:
            f.write(plot_buffer.getbuffer())
        self.assertTrue(os.path.exists(aggregation_plot_path), "Aggregation SHAP plot should be saved to a file")

    def test_explain_privacy_impact(self):
        privacy_params = {'noise_level': 0.5}
        interpretation_text, plot_buffer = self.federated_xai.explain_privacy_impact(self.global_model, self.global_model, privacy_params)
        privacy_impact_plot_path = os.path.join(self.output_dir, "privacy_impact_shap.png")
        with open(privacy_impact_plot_path, "wb") as f:
            f.write(plot_buffer.getbuffer())
        self.assertTrue(os.path.exists(privacy_impact_plot_path), "Privacy impact SHAP plot should be saved to a file")

if __name__ == '__main__':
    unittest.main()
