try:
    from .Elements import Elements
except ImportError:
    from Elements import Elements

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from torch.utils.data import DataLoader, TensorDataset
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import wandb


import torch
import numpy as np
import time
import copy
import logging
import os
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    hamming_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_curve,
    auc
)
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
except ImportError:
    wandb = None
    logging.warning("wandb is not installed. To use wandb logging, please install it: pip install wandb")


class EvaluationUtils:
    @staticmethod
    def _print_bar(value, label, width=25):
        """
        Formats a value and label into a progress bar string.

        Parameters:
            value (float): The value to represent in the bar, expected between 0 and 1.
            label (str): The label for the progress bar.
            width (int): The width of the progress bar.

        Returns:
            str: The formatted progress bar string.
        """
        bar_value = max(0.0, min(1.0, value))
        bar = '=' * int(bar_value * width)
        display_value = f"{value:.4f}"
        return f"{label:<15}: [{bar:<{width}}] {display_value}"

    @staticmethod
    def _calculate_classification_metrics(all_labels_np, all_preds_np, threshold):
        """
        Calculates and returns classification metrics in a dictionary.

        Parameters:
            all_labels_np (np.array): True labels.
            all_preds_np (np.array): Predicted probabilities or logits.
            threshold (float): Threshold for converting probabilities/logits to binary predictions.

        Returns:
            dict: A dictionary containing various classification metrics.
        """
        if not all_labels_np.size or not all_preds_np.size:
            logging.warning("No labels or predictions collected for metric calculation.")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'jaccard_index': 0.0, 'hamming_loss': 1.0
            }

        all_preds_binary = (all_preds_np >= threshold).astype(int)

        accuracy = accuracy_score(all_labels_np, all_preds_binary)
        precision = precision_score(all_labels_np, all_preds_binary, average='samples', zero_division=0)
        recall = recall_score(all_labels_np, all_preds_binary, average='samples', zero_division=0)
        f1 = f1_score(all_labels_np, all_preds_binary, average='samples', zero_division=0)
        jaccard = jaccard_score(all_labels_np, all_preds_binary, average='samples', zero_division=0)
        hamming = hamming_loss(all_labels_np, all_preds_binary)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard_index': jaccard,
            'hamming_loss': hamming
        }

    @staticmethod
    def _calculate_regression_metrics(all_labels_np, all_preds_np):
        """
        Calculates and returns regression metrics in a dictionary.

        Parameters:
            all_labels_np (np.array): True labels.
            all_preds_np (np.array): Predicted values.

        Returns:
            dict: A dictionary containing various regression metrics.
        """
        if not all_labels_np.size or not all_preds_np.size:
            logging.warning("No labels or predictions collected for metric calculation.")
            return {'mse': np.inf, 'mae': np.inf, 'r2_score': -np.inf}

        mse = mean_squared_error(all_labels_np, all_preds_np)
        mae = mean_absolute_error(all_labels_np, all_preds_np)
        r2 = r2_score(all_labels_np, all_preds_np)

        return {'mse': mse, 'mae': mae, 'r2_score': r2}

    @staticmethod
    def train(model, train_loader, valid_loader, optimizer, criterion, epochs=3, mode="supervised", patience=5, threshold=0.5, wandb_run=None):
        """
        Trains a model with options for different modes and evaluates using multiple metrics.

        Parameters:
            model: The neural network model to train.
            train_loader: DataLoader for the training set.
            valid_loader: DataLoader for the validation set.
            optimizer: Optimizer for training (e.g., Adam, SGD).
            criterion: Loss function. Note: For, this should return
                         a tuple (total_loss, reconstruction_loss, classification_loss).
                         For 'regression', this should be a regression loss like MSE or MAE.
            epochs (int): Maximum number of epochs to train.
            mode (str): Training mode. Options: "supervised", "autoencoder", "regression".
            patience (int): Number of epochs to wait for improvement before early stopping.
            threshold (float): Threshold for converting probabilities/logits to binary predictions
                               in supervised mode. Ignored in 'autoencoder' and 'regression' modes.
            wandb_run: Weights & Biases run object for logging.

        Returns:
            dict: A dictionary containing training history, validation history, best validation loss,
                  best model weights (state_dict), and histories of evaluation metrics.
                  The structure of the metrics history depends on the mode.
        """
        best_loss = np.inf
        best_weights = None
        train_history = []
        valid_history = []
        epochs_without_improvement = 0

        metric_histories = {}
        if mode in ["supervised"]:
            metric_histories = {
                'accuracy': [], 'precision': [], 'recall': [],
                'f1_score': [], 'jaccard_index': [], 'hamming_loss': []
            }
        elif mode == "regression":
            metric_histories = {'mse': [], 'mae': [], 'r2_score': []}

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Starting training process (mode: {mode}, epochs: {epochs}, patience: {patience})")
        if mode in ["supervised"]:
            logging.info(f"Threshold for binary prediction: {threshold}")

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            logging.info(f"--- Epoch {epoch + 1}/{epochs} ---")

            total_train_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()

                if mode == "autoencoder":
                    x_pred = model(X_batch)
                    loss = criterion(x_pred, X_batch)
                elif mode in ["supervised", "regression"]:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                else:
                    raise ValueError(f"Invalid mode: {mode}. Choose 'supervised', 'autoencoder' or 'regression'.")

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                if (i + 1) % 10 == 0:
                    logging.info(f"  Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.7f}")

            avg_train_loss = total_train_loss / len(train_loader)
            train_history.append(avg_train_loss)
            logging.info(f"Epoch {epoch + 1} Training Avg Loss: {avg_train_loss:.7f}")
            if wandb_run:
                wandb.log({"train_loss": avg_train_loss}, step=epoch)

            model.eval()
            total_valid_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    if mode == "autoencoder":
                        x_pred = model(X_batch)
                        loss = criterion(x_pred, X_batch)
                    elif mode == "supervised":
                        y_pred = model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        all_preds.extend(y_pred.cpu().numpy())
                        all_labels.extend(y_batch.cpu().numpy())
                    elif mode == "regression":
                        y_pred = model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        all_preds.extend(y_pred.cpu().numpy())
                        all_labels.extend(y_batch.cpu().numpy())
                    else:
                        raise ValueError(f"Invalid mode during validation: {mode}")
                    total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(valid_loader)
            valid_history.append(avg_valid_loss)
            logging.info(f"Epoch {epoch + 1} Validation Avg Loss: {avg_valid_loss:.7f}")
            if wandb_run:
                wandb.log({"val_loss": avg_valid_loss}, step=epoch)

            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)

            if mode in ["supervised"]:
                classification_metrics = EvaluationUtils._calculate_classification_metrics(all_labels_np, all_preds_np, threshold)
                for metric_name, metric_value in classification_metrics.items():
                    metric_histories[metric_name].append(metric_value)

                logging.info(
                    "Validation Metrics:\n"
                    + EvaluationUtils._print_bar(classification_metrics['accuracy'], "Accuracy") + "\n"
                    + EvaluationUtils._print_bar(classification_metrics['precision'], "Precision (s)") + "\n"
                    + EvaluationUtils._print_bar(classification_metrics['recall'], "Recall (s)") + "\n"
                    + EvaluationUtils._print_bar(classification_metrics['f1_score'], "F1 Score (s)") + "\n"
                    + EvaluationUtils._print_bar(classification_metrics['jaccard_index'], "Jaccard Idx (s)") + "\n"
                    + EvaluationUtils._print_bar(1.0 - classification_metrics['hamming_loss'], "1 - Hamm Loss")
                )
                if wandb_run:
                    wandb.log({
                        "val_accuracy": classification_metrics['accuracy'],
                        "val_precision": classification_metrics['precision'],
                        "val_recall": classification_metrics['recall'],
                        "val_f1_score": classification_metrics['f1_score'],
                        "val_jaccard_index": classification_metrics['jaccard_index'],
                        "val_hamming_loss": 1.0 - classification_metrics['hamming_loss']
                    }, step=epoch)
            elif mode == "regression":
                regression_metrics = EvaluationUtils._calculate_regression_metrics(all_labels_np, all_preds_np)
                for metric_name, metric_value in regression_metrics.items():
                    metric_histories[metric_name].append(metric_value)

                logging.info(
                    "Validation Metrics:\n"
                    f"  MSE: {regression_metrics['mse']:.7f}\n"
                    f"  MAE: {regression_metrics['mae']:.7f}\n"
                    f"  R2 Score: {regression_metrics['r2_score']:.7f}"
                )
                if wandb_run:
                    wandb.log({
                        "val_mse": regression_metrics['mse'],
                        "val_mae": regression_metrics['mae'],
                        "val_r2_score": regression_metrics['r2_score']
                    }, step=epoch)

            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_weights = copy.deepcopy(model.state_dict())
                os.makedirs("data/model_weights", exist_ok=True)
                torch.save(model.state_dict(), "data/model_weights/tmp_best_model_weights.pth")
                logging.info(f"Validation loss improved to {best_loss:.4f}. Saving model weights.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logging.info(f"Validation loss did not improve. ({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                logging.warning(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss.")
                break

            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.\n")

        if best_weights is None:
            best_weights = copy.deepcopy(model.state_dict())
            logging.warning("Training finished without improvement; returning last model state.")
        else:
            logging.info(f"Training finished. Best validation loss: {best_loss:.4f}")

        results = {
            'train_loss_history': train_history,
            'valid_loss_history': valid_history,
            'best_valid_loss': best_loss,
            'best_model_weights': best_weights,
            'metric_histories': metric_histories
        }
        return results

    @staticmethod
    def evaluate(model, data_loader, criterion=None, mode="supervised", threshold=0.5):
        """
        Evaluates a trained model on a given dataset.

        Parameters:
            model: The trained neural network model.
            data_loader: DataLoader for the evaluation dataset.
            criterion: Optional. The loss function to calculate evaluation loss.
            mode (str): Evaluation mode. Options: "supervised", "autoencoder","regression".
            threshold (float): Threshold for converting probabilities/logits to binary predictions
                               in supervised mode. Ignored in 'autoencoder' and 'regression' modes.

        Returns:
            dict: A dictionary containing the total loss (if criterion is provided) and evaluation metrics
                  based on the mode.
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        logging.info(f"Starting evaluation process (mode: {mode})")
        if mode in ["supervised"]:
            logging.info(f"Threshold for binary prediction: {threshold}")

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                if mode == "autoencoder":
                    x_pred = model(X_batch)
                    if criterion is not None:
                        loss = criterion(x_pred, X_batch)
                        total_loss += loss.item()
                elif mode == "supervised":
                    y_pred = model(X_batch)
                    if criterion is not None:
                        loss = criterion(y_pred, y_batch)
                        total_loss += loss.item()
                    all_preds.extend(y_pred.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                elif mode == "regression":
                    y_pred = model(X_batch)
                    if criterion is not None:
                        loss = criterion(y_pred, y_batch)
                        total_loss += loss.item()
                    all_preds.extend(y_pred.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                else:
                    raise ValueError(f"Invalid mode during evaluation: {mode}")

        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        evaluation_metrics = {}
        if mode in ["supervised"]:
            evaluation_metrics = EvaluationUtils.calculate_multilabel_metrics(model, data_loader, threshold=threshold)
            logging.info(
                "Evaluation Metrics:\n"
                + EvaluationUtils._print_bar(evaluation_metrics['exact_match_ratio'], "Exact Match Ratio") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['accuracy_label_avg'], "Accuracy (Label Avg)") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['macro_precision'], "Macro Precision") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['macro_recall'], "Macro Recall") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['macro_f1'], "Macro F1 Score") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['macro_jaccard'], "Macro Jaccard Idx") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['micro_precision'], "Micro Precision") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['micro_recall'], "Micro Recall") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['micro_f1'], "Micro F1 Score") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['samples_precision'], "Samples Precision") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['samples_recall'], "Samples Recall") + "\n"
                + EvaluationUtils._print_bar(evaluation_metrics['samples_f1'], "Samples F1 Score") + "\n"
                + EvaluationUtils._print_bar(1.0 - evaluation_metrics['hamming_loss'], "1 - Hamming Loss")
            )
        elif mode == "regression":
            evaluation_metrics = EvaluationUtils._calculate_regression_metrics(all_labels_np, all_preds_np)
            logging.info(
                "Evaluation Metrics:\n"
                f"  MSE: {evaluation_metrics['mse']:.7f}\n"
                f"  MAE: {evaluation_metrics['mae']:.7f}\n"
                f"  R2 Score: {evaluation_metrics['r2_score']:.7f}"
            )

        logging.info("Evaluation process finished.")

        results = {
            'total_loss': total_loss,
            'evaluation_metrics': evaluation_metrics
        }
        return results

    @staticmethod
    def create_dataloader(X, y, device, batch_size=16, shuffle=True):
        """
        Creates a DataLoader from NumPy arrays.

        Parameters:
            X (np.array): Input features.
            y (np.array): Target labels.
            device (torch.device): The device to load tensors onto.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            torch.utils.data.DataLoader: The created DataLoader.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1, X.shape[-1]).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def calculate_roc(model, test_loader):
        """
        Calculates ROC curve and AUC for multi-label classification.

        Parameters:
            model (torch.nn.Module): The trained PyTorch model.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test set.

        Returns:
            tuple: A tuple containing:
                - dict: AUC for each class.
                - dict: FPR and TPR values for each class.
        """
        model.eval()
        all_real, all_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                all_real.append(y_batch.cpu())
                all_pred.append(y_pred.cpu())

        all_real = torch.cat(all_real).numpy()
        all_pred = torch.cat(all_pred).numpy()
        n_classes = all_real.shape[1]
        aucs = {}
        rocs = {}
        for i in range(n_classes):
            if i == len(Elements.NUM2SYMBOL):
                symbol = "wood"
            else:
                symbol = Elements.NUM2SYMBOL[i]
            fpr, tpr, _ = roc_curve(all_real[:, i], all_pred[:, i])
            roc_auc = auc(fpr, tpr)
            aucs[symbol] = roc_auc
            rocs[symbol] = (fpr, tpr)

        return aucs, rocs

    @staticmethod
    def calculate_multilabel_metrics(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu', threshold=0.5):
        """
        Calculates various multi-label classification metrics.

        Parameters:
            model (torch.nn.Module): The trained PyTorch model.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
            device (str): The device to run inference on ('cuda' or 'cpu').
            threshold (float): The threshold to convert model outputs to binary predictions.

        Returns:
            dict: A dictionary containing the calculated metrics:
                - 'exact_match_ratio': (Subset Accuracy) Fraction of samples where prediction perfectly matches true labels.
                - 'hamming_loss': Fraction of incorrect labels over all labels.
                - 'accuracy_label_avg': Average accuracy across all individual labels (original 'accuracy').
                - 'macro_precision': Macro-averaged precision across labels.
                - 'macro_recall': Macro-averaged recall across labels.
                - 'macro_f1': Macro-averaged F1-score across labels.
                - 'macro_jaccard': Macro-averaged Jaccard index (IoU) across labels.
                - 'micro_precision': Precision calculated globally by counting total TPs, FPs.
                - 'micro_recall': Recall calculated globally by counting total TPs, FNs.
                - 'micro_f1': F1-score calculated globally.
                - 'samples_precision': Precision averaged per sample.
                - 'samples_recall': Recall averaged per sample.
                - 'samples_f1': F1-score averaged per sample.
        """
        all_true_np = []
        all_pred_np = []

        model.eval()
        model.to(device)

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = (outputs > threshold).float()

                all_true_np.append(y_batch.cpu().numpy())
                all_pred_np.append(preds.cpu().numpy())

        y_true = np.concatenate(all_true_np, axis=0).astype(int)
        y_pred = np.concatenate(all_pred_np, axis=0).astype(int)

        exact_match_ratio = metrics.accuracy_score(y_true, y_pred)
        hamming_loss_val = metrics.hamming_loss(y_true, y_pred)
        accuracy_label_avg = np.mean(y_pred == y_true) 

        macro_precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        macro_jaccard = metrics.jaccard_score(y_true, y_pred, average='macro', zero_division=0)

        micro_precision = metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)
        micro_recall = metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)
        micro_f1 = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)

        samples_precision = metrics.precision_score(y_true, y_pred, average='samples', zero_division=0)
        samples_recall = metrics.recall_score(y_true, y_pred, average='samples', zero_division=0)
        samples_f1 = metrics.f1_score(y_true, y_pred, average='samples', zero_division=0)

        results = {
            'exact_match_ratio': exact_match_ratio,
            'hamming_loss': hamming_loss_val,
            'accuracy_label_avg': accuracy_label_avg,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'macro_jaccard': macro_jaccard,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'samples_precision': samples_precision,
            'samples_recall': samples_recall,
            'samples_f1': samples_f1,
        }
        return results
