from .Elements import Elements
from .FeatureEnhancer import FeatureEnhancer
from .ArtifficialTrainDataGenerator import ArtifficialTrainDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

class EvaluationUtils:

    @staticmethod
    def train(model, train_loader, valid_loader, optimizer, criterion, epochs=3, mode="supervised", patience=5, threshold=0.5):
        def _print_bar(value, label, width=25):
            bar = '=' * int(value * width)  
            return f"{label:{" "}<15}: [{bar:<{width}}] {value:.4f}"

        best_loss = np.inf
        best_weights = None
        train_history = []
        valid_history = []
        epochs_without_improvement = 0

        # Lists to store metrics
        f1_scores = []
        accuracies = []
        recalls = []
        precisions = []

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            logging.info(f"Epoch {epoch + 1}/{epochs} started.")

            total_train_loss = 0.0
            batch_count = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                if mode == "autoencoder":
                    x_pred = model(X_batch)
                    loss = criterion(x_pred, X_batch)
                elif mode == "supervised":
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                elif mode == "constrained_autoencoder":
                    X_pred, y_pred, z = model(X_batch)
                    loss, x_loss, y_loss = criterion(y_pred, X_pred, y_batch, X_batch)
                else:
                    raise ValueError(f"Invalid mode: {mode}")

                loss.backward()
                optimizer.step()

                batch_count += 1
                total_train_loss += loss.item()
                # if batch_count % 100 == 0:
                #     logging.info(f"Batch {batch_count}, Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / batch_count
            train_history.append(avg_train_loss)
            logging.info(f"Train Epoch Average Loss: {avg_train_loss:.4f}")

            model.eval()
            valid_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    if mode == "autoencoder":
                        x_pred = model(X_batch)
                        valid_loss += criterion(x_pred, X_batch).item()
                    elif mode == "supervised":
                        y_pred = model(X_batch)
                        valid_loss += criterion(y_pred, y_batch).item()

                        # Apply threshold to get binary predictions
                        y_pred_binary = y_pred.round().cpu().numpy()
                        all_preds.extend(y_pred_binary)
                        all_labels.extend(y_batch.cpu().numpy())
                    elif mode == "constrained_autoencoder":
                        X_pred, y_pred, z = model(X_batch)
                        loss, x_loss, y_loss = criterion(y_pred, X_pred, y_batch, X_batch)
                        valid_loss += loss.item()

                        # Apply threshold to get binary predictions
                        y_pred_binary = y_pred.round().cpu().numpy()
                        all_preds.extend(y_pred_binary)
                        all_labels.extend(y_batch.cpu().numpy())
                    else:
                        raise ValueError(f"Invalid mode: {mode}")

            avg_valid_loss = valid_loss / len(valid_loader)
            valid_history.append(avg_valid_loss)
            logging.info(f"Validation Loss: {avg_valid_loss:.4f}")

            # Calculate metrics for multi-label classification
            if mode in ["supervised", "constrained_autoencoder"]:
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, average='samples')
                recall = recall_score(all_labels, all_preds, average='samples')
                f1 = f1_score(all_labels, all_preds, average='samples')

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)


                logging.info("\n" + _print_bar(accuracy, "Accuracy") + "\n" +
                        _print_bar(precision, "Precision") + "\n" +
                        _print_bar(recall, "Recall") + "\n" +
                        _print_bar(f1, "F1 Score"))

            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_weights = copy.deepcopy(model.state_dict())
                logging.info("New best model saved.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch + 1} finished in {epoch_duration:.2f} seconds.\n")

        return train_history, valid_history, best_loss, best_weights, f1_scores, accuracies, recalls, precisions


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
        aucs = []

        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(all_real[:, i], all_pred[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, label=f"{Elements.NUM2SYMBOL[i].capitalize()} (AUC = {roc_auc:.2f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves by Element")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("roc_curves.svg")
        plt.close()

        avg_auc = np.mean(aucs)
        print(f"Average AUC: {avg_auc:.4f}")
        return aucs

    @staticmethod
    def calculate_accuracy_precision_recall_f1(model, test_loader):
        all_true, all_pred = [], []

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = (outputs > 0.5).float()
                all_true.append(y_batch.cpu())
                all_pred.append(preds.cpu())

        all_true = torch.cat(all_true)
        all_pred = torch.cat(all_pred)

        tp = (all_true * all_pred).sum(dim=0)
        fp = ((1 - all_true) * all_pred).sum(dim=0)
        fn = (all_true * (1 - all_pred)).sum(dim=0)

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        avg_precision = precision.mean().item()
        avg_recall = recall.mean().item()
        avg_f1 = f1.mean().item()
        accuracy = (all_pred == all_true).float().mean().item()

        return avg_precision, avg_recall, avg_f1, accuracy