import torch
from src.TrainDataGenerator import TrainDataGenerator
from src.FeatureEnhancer import FeatureEnhancer 
from src.EvaluationUtils import Evaluation
from sklearn.model_selection import train_test_split

X, y = TrainDataGenerator.generate_artificial_data(samples=10, max_elements_per_sample=2, use_max=True, mu_max_err=0.0, mu_max_err_global=0.0, sigma_max_err=0.0, use_percentages=False, cache_element_samples=True)
X = FeatureEnhancer.enhanced_features(X)
X_train, X_valid, y_train, y_valid= train_test_split(X, y, train_size=0.9, shuffle=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
train_loader = Evaluation.create_dataloader(X_train, y_train, DEVICE, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = Evaluation.create_dataloader(X_valid, y_valid, DEVICE, batch_size=BATCH_SIZE, shuffle=True)