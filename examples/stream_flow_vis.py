import warnings
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def visualize_time_series_data_with_streamflow():
    TimeSeriesPlot().usgs_streamflow(path_to_folder='../data/usgs_streamflow')


if __name__ == '__main__':
    visualize_time_series_data_with_streamflow()
