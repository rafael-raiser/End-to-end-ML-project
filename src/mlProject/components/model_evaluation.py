import pandas as pd
import os
from mlProject import logger
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mlProject.config.configuration import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    

    def evaluate(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        predicted_y = np.around(model.predict(test_x)).astype(int)

        cm = pd.DataFrame()
        cm['y_true'] = test_y
        cm['y_predicted'] = predicted_y

        cm.to_csv(self.config.confusion_matrix)

        matrix = confusion_matrix(y_pred=predicted_y, y_true=test_y)

        fig = ConfusionMatrixDisplay(matrix)
        fig.plot()
        plt.savefig(self.config.confusion_matrix_plot)