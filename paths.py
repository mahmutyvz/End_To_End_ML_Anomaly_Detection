class Path:
    """
    This class encapsulates paths and constants related to file locations and dataset properties.

    Attributes:

    root: The root directory path.
    exchange_path: Path to the 'realAdExchange' dataset.
    awscloudwatch_path: Path to the 'realAWSCloudwatch' dataset.
    knowncause_path: Path to the 'realKnownCause' dataset.
    traffic_path: Path to the 'realTraffic' dataset.
    tweets_path: Path to the 'realTweets' dataset.
    timestamp_column: The name of the timestamp column in the datasets.
    random_state: A constant representing the random state for models (set to 42 in this case).
    """
    root = 'C:/Users/MahmutYAVUZ/Desktop/Software/Python/kaggle/anomaly_detection/data/raw/'
    exchange_path = root+'realAdExchange'
    awscloudwatch_path = root+'realAWSCloudwatch'
    knowncause_path = root+'realKnownCause'
    traffic_path = root+'realTraffic'
    tweets_path = root+"realTweets"
    timestamp_column='timestamp'
    random_state = 42