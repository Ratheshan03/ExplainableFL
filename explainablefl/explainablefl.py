import copy
import io
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

class FederatedXAI:
    def __init__(self, device, global_model, data_loader):
        """
        Initialize the FederatedXAI class with a device, a global model, and a data loader.

        Parameters:
        - device (torch.device): The device (CPU/GPU) for computations.
        - global_model (torch.nn.Module): The pre-trained global model.
        - data_loader (torch.utils.data.DataLoader): Common DataLoader for various model explanations.
        """
        self.device = device
        self.global_model = global_model
        self.data_loader = data_loader
        

    def explain_client_model(self, model, num_background=50, num_test_images=14):
        """
        Generates visual explanations for a client's model using SHAP values based on a subset of test images.

        Parameters:
        - model (torch.nn.Module): The client's neural network model.
        - num_background (int): Number of background samples to use for initializing the SHAP explainer.
        - num_test_images (int): Number of test images to explain.

        Returns:
        - BytesIO: A buffer containing the generated SHAP plot image.
        - tuple: A tuple containing arrays of SHAP values and the corresponding test images.
        
        This function uses SHAP to interpret the model's predictions on selected test images, highlighting
        how different features contribute to each prediction. The result includes both the visual plot and
        the data used for the analysis.
        """
        model.to(self.device).eval()
        background_data, _ = next(iter(self.data_loader))
        background_data = background_data[:num_background].to(self.device)
        test_images, _ = next(iter(self.data_loader))
        test_images = test_images[:num_test_images].to(self.device)

        explainer = shap.GradientExplainer(model, background_data)
        shap_values = explainer.shap_values(test_images)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        shap_plot_buf = io.BytesIO()
        plt.figure()
        shap.image_plot(shap_numpy, -test_numpy)
        plt.savefig(shap_plot_buf, format='png')
        shap_plot_buf.seek(0)
        plt.close()

        return shap_plot_buf, (shap_numpy, test_numpy)
    

    def explain_global_model(self, num_background=50, num_test_images=14):
        """
        Generates both a confusion matrix plot and SHAP explanations for the global model based on test data.

        Parameters:
        - num_background (int): The number of background samples for the SHAP explainer initialization.
        - num_test_images (int): The number of test images to be explained.

        Returns:
        - BytesIO: A buffer containing the confusion matrix plot image.
        - BytesIO: A buffer containing the SHAP plot image.
        - tuple: A tuple containing arrays of SHAP values and the corresponding test images.
        
        This method provides insights into the global model's performance and decision-making process, using
        SHAP values to explain how different features influence the model's predictions on the test images.
        """
        self.global_model.to(self.device).eval()
        background_data, _ = next(iter(self.data_loader))
        background_data = background_data[:num_background].to(self.device)
        test_images, _ = next(iter(self.data_loader))
        test_images = test_images[:num_test_images].to(self.device)

        explainer = shap.GradientExplainer(self.global_model, background_data)
        shap_values = explainer.shap_values(test_images)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        shap_plot_buf = io.BytesIO()
        plt.figure()
        shap.image_plot(shap_numpy, -test_numpy)
        plt.savefig(shap_plot_buf, format='png')
        shap_plot_buf.seek(0)
        plt.close()

        conf_matrix = np.zeros((2, 2)) 
        fig_conf_matrix = plt.figure(figsize=(8, 8))
        ax = fig_conf_matrix.add_subplot(111)
        ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
        plt.title("Confusion Matrix")
        buf_conf_matrix = io.BytesIO()
        plt.savefig(buf_conf_matrix, format='png')
        buf_conf_matrix.seek(0)
        plt.close()

        return buf_conf_matrix, shap_plot_buf, (shap_numpy, test_numpy)


    def explain_aggregation(self, pre_aggregated_state, post_aggregated_state, num_background=50, num_features=10):
        """
        Visualizes the effects of aggregation on the global model using SHAP values and evaluates model performance.

        Parameters:
        - pre_aggregated_state (dict): The state dictionary of the global model before aggregation.
        - post_aggregated_state (dict): The state dictionary of the global model after aggregation.
        - num_background (int): The number of background instances for the SHAP explainer.
        - num_features (int): The number of top features to display in the summary plot.

        Returns:
        - BytesIO: A buffer containing the plot image which compares model performance and explanations before and
          after aggregation.
        """
        # Load models with pre and post aggregated states
        model_pre = copy.deepcopy(self.global_model)
        model_pre.load_state_dict(pre_aggregated_state)
        model_pre.to(self.device).eval()

        model_post = copy.deepcopy(self.global_model)
        model_post.load_state_dict(post_aggregated_state)
        model_post.to(self.device).eval()

        # Generate SHAP values for both models
        background, _ = next(iter(self.data_loader))
        background = background[:num_background].to(self.device)
        explainer_pre = shap.GradientExplainer(model_pre, background)
        explainer_post = shap.GradientExplainer(model_post, background)
        shap_values_pre = explainer_pre.shap_values(background)
        shap_values_post = explainer_post.shap_values(background)

        # Reshape SHAP values for summary plot
        shap_values_pre_reshaped = [val.reshape(val.shape[0], -1) for val in shap_values_pre]
        shap_values_post_reshaped = [val.reshape(val.shape[0], -1) for val in shap_values_post]

        # Generate summary plot images for pre and post aggregation
        pre_agg_shap_image = self.capture_shap_summary_plot(shap_values_pre_reshaped, num_features)
        post_agg_shap_image = self.capture_shap_summary_plot(shap_values_post_reshaped, num_features)

        # Calculate accuracy for pre and post aggregation models
        inputs, labels = next(iter(self.data_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        with torch.no_grad():
            outputs_pre = model_pre(inputs)
            _, preds_pre = torch.max(outputs_pre, 1)
            outputs_post = model_post(inputs)
            _, preds_post = torch.max(outputs_post, 1)

        accuracy_pre = accuracy_score(labels.cpu().numpy(), preds_pre.cpu().numpy())
        accuracy_post = accuracy_score(labels.cpu().numpy(), preds_post.cpu().numpy())

        # Create comparison plots
        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        num_examples = min(len(background), 3)
        for i in range(num_examples):
            axs[i, 0].imshow(inputs[i].permute(1, 2, 0).cpu().numpy()) 
            axs[i, 0].set_title(f'Label: {labels[i]}, Pred Pre: {preds_pre[i]}, Pred Post: {preds_post[i]}')
            axs[i, 0].axis('off')

        axs[0, 1].imshow(pre_agg_shap_image)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Global Model Pre-Aggregation')

        axs[1, 1].imshow(post_agg_shap_image)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Global Model Post-Aggregation')

        axs[2, 1].bar(['Pre-Aggregation', 'Post-Aggregation'], [accuracy_pre, accuracy_post])
        axs[2, 1].set_title('Model Accuracy Comparison')
        axs[2, 1].set_ylim(0, 1)

        plt.tight_layout()
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plt.close()
        plot_buffer.seek(0)

        return plot_buffer
    

    def explain_privacy_impact(self, model, private_model, privacy_params, num_test_images=50):
        """
        Generates visualizations to compare the impact of differential privacy on a model.

        Parameters:
        - model (torch.nn.Module): The original model before applying differential privacy.
        - private_model (torch.nn.Module): The model after applying differential privacy.
        - privacy_params (dict): Parameters used for differential privacy.
        - num_test_images (int): The number of test images to use for SHAP explanations.

        Returns:
        - str: A brief summary text of the SHAP values' differences.
        - BytesIO: A buffer containing the visualization plot of SHAP values' differences.
        """
        background, _ = next(iter(self.data_loader))
        background = background[:num_test_images].to(self.device)
        explainer = shap.GradientExplainer(model, background)

        test_images, _ = next(iter(self.data_loader))
        test_images = test_images[:num_test_images].to(self.device)
        shap_values = explainer.shap_values(test_images)
        explainer.model = private_model
        private_shap_values = explainer.shap_values(test_images)

        diff_shap_values = np.abs(np.array(shap_values) - np.array(private_shap_values)).mean(axis=0).flatten()  # Flatten the array

        interpretation_text = (
            f"Mean difference in SHAP values: {np.mean(diff_shap_values):.4f}.\n"
            f"Max difference in SHAP values: {np.max(diff_shap_values):.4f}.\n"
            f"Privacy noise level: {privacy_params.get('noise_level', 'N/A')}.\n"
        )

        feature_indices = np.arange(len(diff_shap_values))
        plt.figure(figsize=(10, 6))
        plt.bar(feature_indices, diff_shap_values)
        plt.xlabel('Feature Index')
        plt.ylabel('Difference in SHAP Value')
        plt.title('Differential Privacy Impact on Model Features')

        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plt.close()
        plot_buffer.seek(0)

        return interpretation_text, plot_buffer


    @staticmethod
    def capture_shap_summary_plot(shap_values, num_features=10):
        """
        Creates a SHAP summary plot for the given SHAP values.
        """
        fig, ax = plt.subplots(1, 1)
        shap.summary_plot(shap_values, plot_type="bar", max_display=num_features, show=False)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image