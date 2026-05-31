class MaskError(Exception):

    """Raised when a time series mask does not exist"""

    def __init__(self, mask_file):
        self.message = f'Mask file {str(mask_file)} does not exist.'
        super().__init__(self.message)

    def __str__(self):
        return self.message


class TimeSeriesError(Exception):

    """Raised when a time series variable is incomplete"""

    def __init__(self, window):
        self.message = f'The last completed window was {window}. Expected {int(1e9):,d}.'
        super().__init__(self.message)

    def __str__(self):
        return self.message


class TrainingGridsError(Exception):

    """Raised when the training grids list is incomplete"""

    def __init__(self, grid_path):
        self.message = f'No training grids were found in {str(grid_path)}.'
        super().__init__(self.message)

    def __str__(self):
        return self.message
