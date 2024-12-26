import copy
import matplotlib.pyplot as plt
import numpy as np
import torch 

from sklearn.metrics import roc_curve, auc
from .TrainDataGenerator import TrainDataGenerator
from .FeatureEnhancer import FeatureEnhancer
from .Elements import Elements 
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset

class Evaluation():
    @staticmethod
    def train(model, train_loader, valid_loader, optimizer, criterion, epochs = 3, physics_inormed_loss=False):
        best_loss = np.inf
        best_weights = None
        train_history = []
        valid_history = []
        for epoch in range(epochs):
            model.train()
            print(f"Epoch {epoch}")
            train_loss = 0
            idx = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                if physics_inormed_loss == False:
                    loss = criterion(y_pred, y_batch)
                else:
                    loss = criterion(y_pred, y_batch, X_batch)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                idx += 1
                train_loss += float(loss)
                if idx % 10 == 0:
                    print(f"Batch {idx}, loss: {float(loss)}")

            train_loss = train_loss/idx
            print("train epoch avg loss: %.2f" % train_loss)
            train_history.append(train_loss)

            with torch.no_grad():
                model.eval()
                def _calc_loss(dataloader):
                    loss_acc = 0
                    for X_batch, y_batch in dataloader:

                        y_pred = model(X_batch)
                        if physics_inormed_loss == False:
                            loss = criterion(y_pred, y_batch)
                        else:
                            loss = criterion(y_pred, y_batch, X_batch)
                        loss_acc += float(loss)
                    return loss_acc / len(dataloader)

                valid_loss = _calc_loss(valid_loader)
                print("valid loss: %.2f" % valid_loss)
                valid_history.append(valid_loss)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_weights = copy.deepcopy(model.state_dict())

        return train_history, valid_history, best_loss, best_weights

    @staticmethod
    def create_dataloader(X, y, device, batch_size=16, shuffle=True):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @staticmethod
    def create_test_dataloader(max_elements_per_sample, device="cpu", samples=10000, use_max=False):
        X, y = TrainDataGenerator.generate_artificial_data(samples=samples
                                                                    , max_elements_per_sample=max_elements_per_sample
                                                                    , use_max=use_max
                                                                    , mu_max_err=0.0
                                                                    , mu_max_err_global=0.0
                                                                    , sigma_max_err=0.0
                                                                    , use_percentages=False
                                                                    , cache_element_samples=True)
        X = FeatureEnhancer.enhanced_features(X)
        return Evaluation.create_dataloader(X
                                        , y
                                        , device=device
                                        , batch_size=128
                                        , shuffle=True)

    @staticmethod
    def calculate_roc(model, test_loader):
        pred_labels = [[] for i in range(TrainDataGenerator.TARGET_VECTOR_LENGTH)]
        real_labels = [[] for i in range(TrainDataGenerator.TARGET_VECTOR_LENGTH)]
        with torch.no_grad():
            model.eval()
            for X_batch, y_batch in test_loader:
                for y in y_batch:
                    for i, label in enumerate(y):
                        real_labels[i].append(label.to("cpu"))

                y_pred = model(X_batch)
                for y in y_pred:
                    for i, label in enumerate(y):
                        pred_labels[i].append(label.to("cpu"))

        aucs = []
        handles_list = []
        labels_list = []
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
        for i in range(TrainDataGenerator.TARGET_VECTOR_LENGTH):
            fpr, tpr, thresholds = roc_curve(real_labels[i], pred_labels[i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            line, = plt.plot(fpr, tpr)

            handles_list.append(line)
            labels_list.append(f"AUC: {roc_auc:.2f} - {Elements.NUM2SYMBOL[i].capitalize()}")

        sorted_lists = sorted(zip(labels_list, handles_list), key = lambda x: x[0], reverse=True)
        sorted_lists = list(zip(*sorted_lists))
        plt.legend(handles=sorted_lists[1], labels=sorted_lists[0], loc='lower right', bbox_to_anchor=(1.5, 0))
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.savefig("vit_equal_classes_roc.svg")

        print(f"Avg auc: {sum(aucs)/len(aucs)}")

    @staticmethod
    def calculate_accuracy_precision_recall_f1(model, test_loader):
        def _precision_recall_f1(y_true, y_pred, epsilon=1e-10):
            true_positives = torch.sum(y_true * y_pred, dim=0)
            false_positives = torch.sum((1 - y_true) * y_pred, dim=0)
            false_negatives = torch.sum(y_true * (1 - y_pred), dim=0)

            precision = true_positives / (true_positives + false_positives + epsilon)
            recall = true_positives / (true_positives + false_negatives + epsilon)

            f1 = 2 * (precision * recall) / (precision + recall + epsilon)

            precision = torch.mean(precision)
            recall = torch.mean(recall)
            f1 = torch.mean(f1)

            return precision.item(), recall.item(), f1.item()

        all_true_labels = []
        all_pred_labels = []
        all_accuracies = []
        with torch.no_grad():
            model.eval()

            for batch in test_loader:
                inputs, true_labels = batch
                outputs = model(inputs)

                pred_labels = (outputs > 0.5).float()

                accuracy = torch.mean((pred_labels == true_labels).float()).item()
                all_accuracies.append(accuracy)

                all_true_labels.append(true_labels)
                all_pred_labels.append(pred_labels)

            all_true_labels = torch.cat(all_true_labels, dim=0)
            all_pred_labels = torch.cat(all_pred_labels, dim=0)

        precision, recall, f1 = _precision_recall_f1(all_true_labels, all_pred_labels)
        accuracy = torch.mean(torch.tensor(all_accuracies)).item()

        return precision, recall, f1, accuracy
