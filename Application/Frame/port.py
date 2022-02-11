import numpy as np

from Utils.log_handler import log_to_console

"""
Module handles the transfer ports for the APPL block
"""


class Port:
    """
    class that describes the port for transferring data
    """

    arr = None

    def __init__(self, name: str, size: int, port_type: str, is_image: bool) -> None:
        """
        Constructor for class port
        Creates a transfer port that is actually a array with the given size and type
        :param name: name of port
        :param size: size in bytes of array
        :param is_image: if port is an image
        :param port_type: type of array the port will be
            'b'	signed char	int	1
            'B'	unsigned char	int	1
            'u'	Py_UNICODE	Unicode	2
            'h'	signed short	int	2
            'H'	unsigned short	int	2
            'i'	signed int	int	2
            'I'	unsigned int	int	2
            'l'	signed long	int	4
            'L'	unsigned long	int	4
            'f'	float	float	4
            'd'	double	float	8
        """
        self.name = name
        self.isValid = False

        eq_dict = {'b': 'np.byte',
                   'B': 'np.ubyte',
                   'h': 'np.short',
                   'H': 'np.ushort',
                   'i': 'np.intc',
                   'I': 'np.uintc',
                   'l': 'np.int_',
                   'L': 'np.uint',
                   'f': 'np.single',
                   'd': 'np.double',
                   'u': 'np.unicode_'}
        self.arr = np.zeros(size, dtype=eval(eq_dict[port_type]))
        self.is_image = is_image

        log_to_console("PORT: {port:150s} is INITIALIZED with SIZE: {size} and CHANNELS: {channel}".
                       format(port=name, size=self.arr.size, channel=len(self.arr.shape)))

    def set_valid(self) -> None:
        """
        Sets the port to valid
        :return: None
        """
        self.isValid = True

    def set_invalid(self) -> None:
        """
        Sets the port to invalid
        :return: None
        """
        self.isValid = False

    def reset_valid_flag(self) -> None:
        """
        Resets the valid flag of the port
        :return: None
        """
        self.isValid = False

    def is_valid(self) -> bool:
        """
        :return: validity of port
        """
        return self.isValid

    def get_name(self) -> str:
        """
        :return: Name of port
        """
        return self.name

    def get_is_image(self):
        """
        :return: If port is image
        """
        return self.is_image

    def reshape_arr(self, size_new_array: tuple, type_new_array: str) -> None:
        """
        Reshape arr of port
        :param size_new_array: shape tuple for new port
        :param type_new_array: type of shape
        :return: none
        """
        self.arr = np.zeros(size_new_array, dtype=type_new_array)

    def self_reset(self):
        """
        Reset arr of port
        :return: none
        """
        self.arr = np.zeros(shape=self.arr.shape, dtype=self.arr.dtype)
