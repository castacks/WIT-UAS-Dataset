import datetime
import os
import sys

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:  # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            )
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
