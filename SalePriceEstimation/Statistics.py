import pandas as pd
import matplotlib.pyplot as plt


class Statistics:
    """Displays statistics about the content of a pandas DataFrame"""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def show_label_statistics(self, label: str, plot_kind: str = "line") -> None:
        """Displays statistics for a specified label"""
        feature_data = self.df[label]
        feature_stats = feature_data.describe()

        print(f"\n\nStatistics for {label}:")
        print(f"Value counts:\n{feature_data.value_counts()}")

        print(f"dtype: {feature_data.dtype}")
        if feature_data.dtype != object:
            plt.figure()
            plt.legend(loc="best")
            feature_data.plot(kind=plot_kind)
            plt.plot([], [], " ", label=f"min: {feature_stats['min']}")
            plt.plot([], [], " ", label=f"max: {feature_stats['max']}")
            plt.plot([], [], " ", label=f"mean: {feature_stats['mean']}")
            plt.legend()
            plt.show()

    def show_missing_data_statistics(self) -> None:
        print("Missing value by label:")
        print(f"{self.df.isnull().sum()/ len(self.df)}%")

    def show_info(self) -> None:
        print(self.df.info())


if __name__ == "__main__":
    data_frame = pd.read_csv("data/TrainAndValid.csv")
    statistics: Statistics = Statistics(data_frame)
    statistics.show_label_statistics(label="SalePrice", plot_kind="box")
    statistics.show_label_statistics(label="YearMade", plot_kind="box")
    statistics.show_label_statistics(label="MachineHoursCurrentMeter")
    statistics.show_label_statistics(label="SalesID", plot_kind="box")
    statistics.show_missing_data_statistics()
    statistics.show_info()
