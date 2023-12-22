
"""
Base model for text spotters
"""

class TextSpotter:
    def __init__(self):
        raise NotImplementedError("Subclasses must implement the init method")
        

    def predict(self, image_path, out_file=""):
        """
        Placeholder for the predict function.
        This function should be overridden by subclasses.

        Parameters:
        - image_path: The image path for making predictions.
        - out_file: File for storing predictions

        Returns:
        - prediction: The prediction made by the model.
        """
        raise NotImplementedError("Subclasses must implement the predict method")
