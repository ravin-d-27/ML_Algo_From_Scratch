class MeanSquaredError:
    def MSE(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def r2_score(self, y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot)
