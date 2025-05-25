from .ArtifficialTrainDataGenerator import ArtifficialTrainDataGenerator
from .Elements import Elements
from .FeatureEnhancer import FeatureEnhancer
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, jaccard_score, hamming_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Import regression metrics
from torch.utils.data import DataLoader, TensorDataset
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import wandb


class EvaluationUtils:

    @staticmethod
    def train(model, train_loader, valid_loader, optimizer, criterion, epochs=3, mode="supervised", patience=5, threshold=0.5, wandb_run=None):
        """
        Trains a model with options for different modes and evaluates using multiple metrics.

        Args:
            model: The neural network model to train.
            train_loader: DataLoader for the training set.
            valid_loader: DataLoader for the validation set.
            optimizer: Optimizer for training (e.g., Adam, SGD).
            criterion: Loss function. Note: For 'constrained_autoencoder', this should return
                         a tuple (total_loss, reconstruction_loss, classification_loss).
                         For 'regression', this should be a regression loss like MSE or MAE.
            epochs (int): Maximum number of epochs to train.
            mode (str): Training mode. Options: "supervised", "autoencoder", "constrained_autoencoder", "regression".
            patience (int): Number of epochs to wait for improvement before early stopping.
            threshold (float): Threshold for converting probabilities/logits to binary predictions
                               in supervised and constrained_autoencoder modes. Ignored in 'autoencoder' and 'regression' modes.

        Returns:
            dict: A dictionary containing training history, validation history, best validation loss,
                  best model weights (state_dict), and histories of evaluation metrics.
                  The structure of the metrics history depends on the mode.
        """

        # --- Helper Functions ---
        def _print_bar(value, label, width=25):
            """Formats a value and label into a progress bar string."""
            # Ensure value is within [0, 1] for bar representation, clamp otherwise
            bar_value = max(0.0, min(1.0, value))
            bar = '=' * int(bar_value * width)
            # Handle cases where value might be outside [0, 1] conceptually
            display_value = f"{value:.4f}"
            return f"{label:{" "}<15}: [{bar:<{width}}] {display_value}"

        def _calculate_classification_metrics(all_labels_np, all_preds_np, threshold):
            """Calculates and returns classification metrics in a dictionary."""
            if not all_labels_np.size or not all_preds_np.size:
                logging.warning("No labels or predictions collected for metric calculation.")
                if  wandb_run:
                    wandb.log({"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "jaccard_index": 0, "hamming_loss": 1})
                return {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'f1_score': 0.0, 'jaccard_index': 0.0, 'hamming_loss': 1.0
                }

            # Ensure predictions are binary based on the threshold
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

        def _calculate_regression_metrics(all_labels_np, all_preds_np):
            """Calculates and returns regression metrics in a dictionary."""
            if not all_labels_np.size or not all_preds_np.size:
                 logging.warning("No labels or predictions collected for metric calculation.")
                 return {'mse': np.inf, 'mae': np.inf, 'r2_score': -np.inf}

            mse = mean_squared_error(all_labels_np, all_preds_np)
            mae = mean_absolute_error(all_labels_np, all_preds_np)
            r2 = r2_score(all_labels_np, all_preds_np)

            return {'mse': mse, 'mae': mae, 'r2_score': r2}


        best_loss = np.inf
        best_weights = None
        train_history = []
        valid_history = []
        epochs_without_improvement = 0

        # Metric histories (initialized based on mode)
        metric_histories = {}
        if mode in ["supervised", "constrained_autoencoder"]:
            metric_histories = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'jaccard_index': [],
                'hamming_loss': []
            }
        elif mode == "regression":
             metric_histories = {
                'mse': [],
                'mae': [],
                'r2_score': []
             }


        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Starting training process (mode: {mode}, epochs: {epochs}, patience: {patience})")
        if mode in ["supervised", "constrained_autoencoder"]:
             logging.info(f"Threshold for binary prediction: {threshold}")

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            logging.info(f"--- Epoch {epoch + 1}/{epochs} ---")

            total_train_loss = 0.0
            batch_count = 0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()

                loss = torch.tensor(0.0) # Initialize loss for the batch

                if mode == "autoencoder":
                    x_pred = model(X_batch)
                    loss = criterion(x_pred, X_batch)
                elif mode == "supervised":
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                elif mode == "regression":
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                elif mode == "constrained_autoencoder":
                    X_pred, y_pred, z = model(X_batch)
                    loss_tuple = criterion(y_pred, X_pred, y_batch, X_batch)
                    if isinstance(loss_tuple, tuple):
                        loss, x_loss, y_loss = loss_tuple
                    else:
                        loss = loss_tuple # Assuming total loss is returned if not a tuple
                else:
                    raise ValueError(f"Invalid mode: {mode}. Choose 'supervised', 'autoencoder', 'constrained_autoencoder', or 'regression'.")

                loss.backward()
                optimizer.step()

                batch_count += 1
                total_train_loss += loss.item()
                # Optional: Log batch loss periodically
                if (i + 1) % 10 == 0:
                    logging.info(f"  Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.7f}")

            avg_train_loss = total_train_loss / batch_count
            train_history.append(avg_train_loss)
            logging.info(f"Epoch {epoch + 1} Training Avg Loss: {avg_train_loss:.7f}")

            model.eval()
            total_valid_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:

                    if mode == "autoencoder":
                        x_pred = model(X_batch)
                        loss = criterion(x_pred, X_batch)
                        total_valid_loss += loss.item()

                    elif mode in ["supervised", "regression", "constrained_autoencoder"]:
                         if mode == "supervised":
                             y_pred = model(X_batch)
                             loss = criterion(y_pred, y_batch)
                         elif mode == "regression":
                             y_pred = model(X_batch)
                             loss = criterion(y_pred, y_batch)
                         elif mode == "constrained_autoencoder":
                             X_pred, y_pred, z = model(X_batch)
                             loss_tuple = criterion(y_pred, X_pred, y_batch, X_batch)
                             if isinstance(loss_tuple, tuple):
                                 loss, _, _ = loss_tuple # Only need total loss for validation loss tracking
                             else:
                                 loss = loss_tuple # Assuming total loss is returned

                         total_valid_loss += loss.item()

                         # Collect predictions and labels for metrics
                         all_preds.extend(y_pred.cpu().numpy())
                         all_labels.extend(y_batch.cpu().numpy())

                    else:
                         raise ValueError(f"Invalid mode during validation: {mode}")

            avg_valid_loss = total_valid_loss / len(valid_loader)
            valid_history.append(avg_valid_loss)
            logging.info(f"Epoch {epoch + 1} Validation Avg Loss: {avg_valid_loss:.7f}")

            # Calculate and log metrics based on mode
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)

            if mode in ["supervised", "constrained_autoencoder"]:
                classification_metrics = _calculate_classification_metrics(all_labels_np, all_preds_np, threshold)
                for metric_name, metric_value in classification_metrics.items():
                    metric_histories[metric_name].append(metric_value)
                
                logging.info("Validation Metrics:\n" +
                            _print_bar(classification_metrics['accuracy'], "Accuracy") + "\n" +
                            _print_bar(classification_metrics['precision'], "Precision (s)") + "\n" +
                            _print_bar(classification_metrics['recall'], "Recall (s)") + "\n" +
                            _print_bar(classification_metrics['f1_score'], "F1 Score (s)") + "\n" +
                            _print_bar(classification_metrics['jaccard_index'], "Jaccard Idx (s)") + "\n" +
                            _print_bar(1.0 - classification_metrics['hamming_loss'], "1 - Hamm Loss") # Display 1 - Hamming Loss for easier interpretation
                        )
                if  wandb_run:
                    wandb.log({
                        "val_accuracy": classification_metrics['accuracy'],
                        "val_precision": classification_metrics['precision'],
                        "val_recall": classification_metrics['recall'],
                        "val_f1_score": classification_metrics['f1_score'],
                        "val_jaccard_index": classification_metrics['jaccard_index'],
                        "val_hamming_loss": 1.0 - classification_metrics['hamming_loss']
                    })

            elif mode == "regression":
                 regression_metrics = _calculate_regression_metrics(all_labels_np, all_preds_np)
                 for metric_name, metric_value in regression_metrics.items():
                     metric_histories[metric_name].append(metric_value)

                 logging.info("Validation Metrics:\n" +
                              f"  MSE: {regression_metrics['mse']:.7f}\n" +
                              f"  MAE: {regression_metrics['mae']:.7f}\n" +
                              f"  R2 Score: {regression_metrics['r2_score']:.7f}"
                             )
                
                 if  wandb_run:
                    wandb.log({
                        "val_mse": regression_metrics['mse'],
                        "val_mae": regression_metrics['mae'],
                        "val_r2_score": regression_metrics['r2_score']
                    })


            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_weights = copy.deepcopy(model.state_dict())
                # Consider saving to a more specific filename or using a temporary file approach
                # For simplicity, keeping the original logic but be mindful in production
                try:
                    # Ensure the data/model_weights directory exists
                    import os
                    os.makedirs("data/model_weights", exist_ok=True)
                    torch.save(model.state_dict(), "data/model_weights/tmp_best_model_weights.pth")
                except Exception as e:
                    logging.warning(f"Could not save model weights to data/model_weights/tmp_best_model_weights.pth: {e}")

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
             # If no improvement was ever seen (e.g., first epoch had issues), return the last state
             best_weights = copy.deepcopy(model.state_dict())
             logging.warning("Training finished without improvement; returning last model state.")
        else:
             logging.info(f"Training finished. Best validation loss: {best_loss:.4f}")

        # Prepare the return dictionary
        results = {
            'train_loss_history': train_history,
            'valid_loss_history': valid_history,
            'best_valid_loss': best_loss,
            'best_model_weights': best_weights,
            'metric_histories': metric_histories # This already contains mode-specific metrics
        }

        return results


    @staticmethod
    def evaluate(model, data_loader, mode="supervised", threshold=0.5):
        """
        Evaluates a trained model on a given dataset.

        Args:
            model: The trained neural network model.
            data_loader: DataLoader for the evaluation dataset.
            mode (str): Evaluation mode. Options: "supervised", "autoencoder", "constrained_autoencoder", "regression".
            threshold (float): Threshold for converting probabilities/logits to binary predictions
                               in supervised and constrained_autoencoder modes. Ignored in 'autoencoder' and 'regression' modes.
            criterion: Optional. The loss function to calculate evaluation loss.

        Returns:
            dict: A dictionary containing the total loss (if criterion is provided) and evaluation metrics
                  based on the mode.
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Criterion is not passed in evaluate, assuming loss calculation is part of the model or not needed for evaluation metrics only.
        # If loss is needed, the criterion would need to be passed or inferred.
        # For now, focusing on metrics calculable from preds and labels/inputs.
        criterion = None # Placeholder - pass criterion to this method if needed for evaluation loss.


        logging.info(f"Starting evaluation process (mode: {mode})")
        if mode in ["supervised", "constrained_autoencoder"]:
             logging.info(f"Threshold for binary prediction: {threshold}")


        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                if mode == "autoencoder":
                    x_pred = model(X_batch)
                    if criterion:
                        loss = criterion(x_pred, X_batch)
                        total_loss += loss.item()
                    # No additional metrics collected for standard autoencoder evaluation

                elif mode in ["supervised", "regression", "constrained_autoencoder"]:
                    if mode == "supervised":
                        y_pred = model(X_batch)
                        if criterion:
                             loss = criterion(y_pred, y_batch)
                             total_loss += loss.item()
                    elif mode == "regression":
                        y_pred = model(X_batch)
                        if criterion:
                             loss = criterion(y_pred, y_batch)
                             total_loss += loss.item()
                    elif mode == "constrained_autoencoder":
                        X_pred, y_pred, z = model(X_batch)
                        if criterion:
                             loss_tuple = criterion(y_pred, X_pred, y_batch, X_batch)
                             if isinstance(loss_tuple, tuple): loss, _, _ = loss_tuple
                             else: loss = loss_tuple
                             total_loss += loss.item()

                    # Collect predictions and labels for metrics
                    all_preds.extend(y_pred.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

                else:
                    raise ValueError(f"Invalid mode during evaluation: {mode}")

        # Calculate and return metrics based on mode
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        evaluation_metrics = {}
        if mode in ["supervised", "constrained_autoencoder"]:
            evaluation_metrics = EvaluationUtils._calculate_classification_metrics(all_labels_np, all_preds_np, threshold)
            logging.info("Evaluation Metrics:\n" +
                         EvaluationUtils._print_bar(evaluation_metrics['accuracy'], "Accuracy") + "\n" +
                         EvaluationUtils._print_bar(evaluation_metrics['precision'], "Precision (s)") + "\n" +
                         EvaluationUtils._print_bar(evaluation_metrics['recall'], "Recall (s)") + "\n" +
                         EvaluationUtils._print_bar(evaluation_metrics['f1_score'], "F1 Score (s)") + "\n" +
                         EvaluationUtils._print_bar(evaluation_metrics['jaccard_index'], "Jaccard Idx (s)") + "\n" +
                         EvaluationUtils._print_bar(1.0 - evaluation_metrics['hamming_loss'], "1 - Hamm Loss")
                       )

        elif mode == "regression":
            evaluation_metrics = EvaluationUtils._calculate_regression_metrics(all_labels_np, all_preds_np)
            logging.info("Evaluation Metrics:\n" +
                         f"  MSE: {evaluation_metrics['mse']:.7f}\n" +
                         f"  MAE: {evaluation_metrics['mae']:.7f}\n" +
                         f"  R2 Score: {evaluation_metrics['r2_score']:.7f}"
                        )
        # If mode is autoencoder, evaluation_metrics remains empty

        logging.info("Evaluation process finished.")

        # Prepare the return dictionary
        results = {
            'total_loss': total_loss, # Will be 0.0 if criterion is not provided
            'evaluation_metrics': evaluation_metrics # Contains mode-specific metrics or is empty for autoencoder
        }

        return results



    @staticmethod
    def create_dataloader(X, y, device, batch_size=16, shuffle=True):
        X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1, X.shape[-1]).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def create_test_dataloader(max_elements_per_sample, device="cpu", samples=10000, use_max=False):
        X, y = ArtifficialTrainDataGenerator.generate_artificial_data(
            samples=samples,
            max_elements_per_sample=max_elements_per_sample,
            use_max=use_max,
            mu_max_err=0.0,
            mu_max_err_global=0.0,
            sigma_max_err=0.0,
            use_percentages=False,
            cache_element_samples=True
        )
        X = FeatureEnhancer.process_spectra(X)
        return EvaluationUtils.create_dataloader(
            X, y, device=device, batch_size=128, shuffle=True
        )

    @staticmethod
    def calculate_roc(model, test_loader):
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

        Args:
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
                - 'samples_f1': F1-score averaged per sample. (Optional, but often useful)
        """
        all_true_np = []
        all_pred_np = []

        model.eval()
        model.to(device) # Ensure model is on the correct device

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                # y_batch stays on CPU or move labels to device if needed by loss/logic
                # For metrics calculation, CPU is fine as we collect and use sklearn later

                outputs = model(X_batch)
                # Assuming outputs are probabilities/logits per label (after sigmoid if needed)
                preds = (outputs > threshold).float()

                all_true_np.append(y_batch.cpu().numpy())
                all_pred_np.append(preds.cpu().numpy())

        # Concatenate all batches
        # Use numpy concatenation as sklearn metrics work best with numpy arrays
        y_true = np.concatenate(all_true_np, axis=0)
        y_pred = np.concatenate(all_pred_np, axis=0)

        # Ensure integer type for sklearn metrics (often required or preferred)
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        # --- Calculate Metrics using scikit-learn ---

        # 1. Exact Match Ratio (Subset Accuracy)
        # Fraction of samples that have all their labels classified correctly.
        exact_match_ratio = metrics.accuracy_score(y_true, y_pred)

        # 2. Hamming Loss
        # The fraction of labels that are incorrectly predicted. Lower is better.
        # (Number of incorrect labels) / (Total number of labels)
        hamming_loss_val = metrics.hamming_loss(y_true, y_pred)

        # 3. Precision, Recall, F1-Score (Macro, Micro, Samples)
        # Use zero_division=0 to avoid warnings and return 0 when precision/recall is undefined (e.g., no true positives and no false positives/negatives)
        macro_precision = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)

        micro_precision = metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)
        micro_recall = metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)
        micro_f1 = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)

        # F1 averaged per sample (useful if sample performance varies greatly)
        samples_f1 = metrics.f1_score(y_true, y_pred, average='samples', zero_division=0)

        # 4. Jaccard Index (Intersection over Union - IoU)
        # Macro-averaged Jaccard Score
        macro_jaccard = metrics.jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        # Micro-averaged Jaccard Score (often similar to Micro-F1)
        # micro_jaccard = metrics.jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        # Sample-averaged Jaccard Score
        # samples_jaccard = metrics.jaccard_score(y_true, y_pred, average='samples', zero_division=0)


        # --- Original Accuracy Calculation (Label Averaged Accuracy) ---
        # This calculates (TP+TN) / (TP+TN+FP+FN) for each label and averages.
        # Or equivalently, the proportion of correct predictions across all sample-label pairs.
        # (Can be different from Exact Match Ratio and Hamming Loss complement)
        # Replicate using numpy for consistency:
        accuracy_label_avg = np.mean(y_pred == y_true) # Element-wise comparison, then mean


        results = {
            'exact_match_ratio': exact_match_ratio,
            'hamming_loss': hamming_loss_val,
            'accuracy_label_avg': accuracy_label_avg, # The 'accuracy' from original code
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'macro_jaccard': macro_jaccard,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'samples_f1': samples_f1,
        }

        return results