from timeit import default_timer as timer
import sys

"""
Module handles the time object for the APPL block
"""


# noinspection PyTypeChecker,PyTypeChecker
class Timer:
    """
    class for describing the system timer
    """
    CONVERT_TO_MS = 1000

    def __init__(self) -> object:
        """
        Constructor of Timer class
        """
        self.__average_time_sum__: float = 0.0
        self.__average_time_count__: float = 0.0
        self.__average_time__: float = 0.0
        self.__current_time__: float = 0.0
        self.__cycle_start__: float = 0.0
        self.__cycle_end__: float = 0.0

    def start_cycle_timer(self) -> None:
        """
        Sets the timer value for start of cycle
        :return: None
        """
        self.__cycle_start__ = timer()

    def end_cycle_timer(self) -> None:
        """
        Sets the timer value for end of cycle
        :return: None
        """
        self.__cycle_end__ = timer()

    def cycle_updater(self) -> None:
        """
        Updates the cycle time
        :return: None
        """
        # overlaps
        if self.__cycle_end__ < self.__cycle_start__:
            self.__current_time__ = (sys.float_info.max - self.__cycle_start__) + self.__cycle_end__
        else:
            self.__current_time__ = self.__cycle_end__ - self.__cycle_start__

        self.__average_time_count__ += 1
        self.__average_time_sum__ += self.__current_time__
        self.__average_time__ = self.__average_time_sum__ / self.__average_time_count__

    def get_current_time(self) -> float:
        """
        :return: The value in milliseconds of cycle
        """
        return self.__current_time__ * self.CONVERT_TO_MS

    def get_average_time(self) -> float:
        """
        :return: The value in milliseconds of cycle average
        """
        return self.__average_time__ * self.CONVERT_TO_MS

    def get_average_time_seconds(self) -> float:
        """
        :return: The value in milliseconds of cycle average
        """
        return self.__average_time__
