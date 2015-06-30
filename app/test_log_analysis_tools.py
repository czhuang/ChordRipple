
from log_analysis_tools import *
from test_Logs import load_log


def test_log_diversity():
    logs = load_log()
    for log in logs.use:
        print log
    log_analyzer = LogAnalyzer(logs)
    log_analyzer.log_diversity()


if __name__ == '__main__':
    test_log_diversity()

