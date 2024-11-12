class DataSaver():
    """
    Store the generated data in a file of a specified format.
    """
    
    def __init__(self, **kwargs) -> None:
        self.process_type = kwargs["process_type"] # i.e., "hawkes", "mjp"
        self.format = kwargs["format"] # i.e., "h5"