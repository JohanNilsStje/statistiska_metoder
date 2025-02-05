import numpy as np
import scipy.stats as stats

class MultipleLinearRegression:
    def __init__(self, data_file, confidence_level=0.95):
        self.data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
        self.y = self.data[:, 1]
        self.X = np.column_stack((np.ones(len(self.data)), self.data[:, 2:]))
        self._confidence_level = confidence_level
        self.beta = self._calculate_beta()
        self.SSE = self._calculate_SSE()
        self.se_beta = self._calculate_se_beta()

    @property
    def d(self):
        return self.X.shape[1] - 1

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def confidence_level(self):
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, value):
        if 0 < value < 1:
            self._confidence_level = value
        else:
            raise ValueError("Confidence level must be between 0 and 1")

    def _calculate_beta(self):
        return np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y

    def _calculate_SSE(self):
        y_pred = self.X @ self.beta
        return np.sum((self.y - y_pred)**2)

    def calculate_variance(self):
        return self.SSE / (self.n - self.d - 1)

    def calculate_std_dev(self):
        return np.sqrt(self.calculate_variance())

    def _calculate_se_beta(self):
        var_beta = self.calculate_variance() * np.linalg.inv(self.X.T @ self.X)
        return np.sqrt(np.diag(var_beta))

    def report_significance(self):
        F_statistic = self.calculate_f_statistic()
        F_p_value = 1 - stats.f.cdf(F_statistic, self.d, self.n - self.d - 1)
        return F_statistic, F_p_value

    def calculate_r_squared(self):
        SST = np.sum((self.y - np.mean(self.y))**2)
        return 1 - self.SSE / SST

    def individual_significance_tests(self):
        t_stats = self.beta / self.se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.n - self.d - 1))
        return t_stats, p_values

    def calculate_pearson_correlation(self):
        correlation_matrix = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                correlation_matrix[i, j], _ = stats.pearsonr(self.X[:, i+1], self.X[:, j+1])
        return correlation_matrix

    def calculate_confidence_intervals(self):
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, self.n - self.d - 1)
        lower = self.beta - t_value * self.se_beta
        upper = self.beta + t_value * self.se_beta
        return lower, upper

    def calculate_f_statistic(self):
        SSR = np.sum((self.X @ self.beta - np.mean(self.y))**2)
        return (SSR / self.d) / (self.SSE / (self.n - self.d - 1))

    def run_analysis(self):
        print(f"Number of features (d): {self.d}")
        print(f"Sample size (n): {self.n}")
        print(f"Variance: {self.calculate_variance()}")
        print(f"Standard Deviation: {self.calculate_std_dev()}")
        
        F_stat, F_p_value = self.report_significance()
        print(f"F-statistic: {F_stat}, p-value: {F_p_value}")
        
        print(f"R-squared: {self.calculate_r_squared()}")
        
        t_stats, p_values = self.individual_significance_tests()
        print("Individual significance tests:")
        for i, (t, p) in enumerate(zip(t_stats, p_values)):
            print(f"  Feature {i}: t-statistic = {t}, p-value = {p}")
        
        print("Pearson correlation matrix:")
        print(self.calculate_pearson_correlation())
        
        lower, upper = self.calculate_confidence_intervals()
        print("Confidence intervals:")
        for i, (l, u) in enumerate(zip(lower, upper)):
            print(f"  Feature {i}: ({l}, {u})")

# Usage
model = MultipleLinearRegression('Small-diameter-flow.csv')
model.run_analysis()