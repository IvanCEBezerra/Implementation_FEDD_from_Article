from river import stream
import numpy as np


def vector_stream(vector):
    """Creates a stream from a vector. 
    With river, a stream is an iterable of data points. This function takes a vector (a list or numpy array) and yields each value in the vector one at a time.

    Parameters
    ----------
    vector
        A vector of values.

    Yields
    ------
    The values in the vector, one at a time."""

    vector = np.asarray(vector)  # Convert the input to a numpy array for consistency

    reshaped_vector = vector.reshape(-1, 1)  # Reshape the vector to a fake 1D array

    data_stream = stream.iter_array(reshaped_vector)  # Create a stream from the reshaped vector

    for observations in data_stream:
        yield observations  # Yield each observation in the stream