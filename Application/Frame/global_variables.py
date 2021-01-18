# number of picture, default value it will be updated.
import math

from Application.Frame.job import JobState
from Utils.log_handler import log_to_console, log_error_to_console

"""
Module handles the global variables for the APPL block
"""

# Function for returning the state of JobState
# noinspection PyPep8
JobInitStateReturn = lambda v: JobState.INIT if v is True else JobState.NOT_INIT


# noinspection PyUnresolvedReferences
class global_var_handler:
    """
    Class for handling global variables for APPL block
    """

    def __init__(self):
        """
        Constructor of class
        :param self
        :return: None
        """
        global_var_handler.NR_PICTURES = 0
        global_var_handler.FRAME = 0
        global_var_handler.PICT_NAME = None
        global_var_handler.WIDTH_L0 = 500
        global_var_handler.HEIGHT_L0 = 500
        global_var_handler.L0_SIZE = 0
        global_var_handler.HEIGHT_L1 = 0
        global_var_handler.WIDTH_L1 = 0
        global_var_handler.L1_SIZE = 0
        global_var_handler.HEIGHT_L2 = 0
        global_var_handler.WIDTH_L2 = 0
        global_var_handler.L2_SIZE = 0
        global_var_handler.HEIGHT_L3 = 0
        global_var_handler.WIDTH_L3 = 0
        global_var_handler.L3_SIZE = 0
        global_var_handler.HEIGHT_L4 = 0
        global_var_handler.WIDTH_L4 = 0
        global_var_handler.L4_SIZE = 0
        global_var_handler.HEIGHT_L4 = 0
        global_var_handler.WIDTH_L4 = 0
        global_var_handler.L4_SIZE = 0
        global_var_handler.HEIGHT_L5 = 0
        global_var_handler.WIDTH_L5 = 0
        global_var_handler.L5_SIZE = 0
        global_var_handler.HEIGHT_L6 = 0
        global_var_handler.WIDTH_L6 = 0
        global_var_handler.L6_SIZE = 0
        global_var_handler.HEIGHT_L7 = 0
        global_var_handler.WIDTH_L7 = 0
        global_var_handler.L7_SIZE = 0
        global_var_handler.HEIGHT_L8 = 0
        global_var_handler.WIDTH_L8 = 0
        global_var_handler.L8_SIZE = 0

        global_var_handler.STR_L0_SIZE = ''

        global_var_handler.VIDEO = None

        log_to_console('GLOBAL VARIABLES INITIALISED')

    @classmethod
    def recalculate_pyramid_level_values(cls):
        """
        Image size has changed so we need to recalculate all pyramid levels
        :param cls
        :return: None
        """
        global_var_handler.L0_SIZE_RGB = (global_var_handler.HEIGHT_L0, global_var_handler.WIDTH_L0, 3)
        global_var_handler.L0_SIZE = (global_var_handler.HEIGHT_L0, global_var_handler.WIDTH_L0)
        global_var_handler.SIZE_ARRAY = [global_var_handler.L0_SIZE]
        
        global_var_handler.HEIGHT_L1 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 1))
        global_var_handler.WIDTH_L1 = int(math.ceil(global_var_handler.WIDTH_L0 >> 1))
        global_var_handler.L1_SIZE_RGB = (global_var_handler.HEIGHT_L1, global_var_handler.WIDTH_L1, 3)
        global_var_handler.L1_SIZE = (global_var_handler.HEIGHT_L1, global_var_handler.WIDTH_L1)
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L1_SIZE)
        
        global_var_handler.HEIGHT_L2 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 2))
        global_var_handler.WIDTH_L2 = int(math.ceil(global_var_handler.WIDTH_L0 >> 2))
        global_var_handler.L2_SIZE_RGB = (global_var_handler.HEIGHT_L2, global_var_handler.WIDTH_L2, 3)
        global_var_handler.L2_SIZE = (global_var_handler.HEIGHT_L2, global_var_handler.WIDTH_L2)
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L2_SIZE)
        
        global_var_handler.HEIGHT_L3 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 3))
        global_var_handler.WIDTH_L3 = int(math.ceil(global_var_handler.WIDTH_L0 >> 3))
        global_var_handler.L3_SIZE_RGB = (global_var_handler.HEIGHT_L3, global_var_handler.WIDTH_L3, 3)
        global_var_handler.L3_SIZE = (global_var_handler.HEIGHT_L3, global_var_handler.WIDTH_L3)
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L3_SIZE)
        
        global_var_handler.HEIGHT_L4 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 4))
        global_var_handler.WIDTH_L4 = int(math.ceil(global_var_handler.WIDTH_L0 >> 4))
        global_var_handler.L4_SIZE_RGB = (global_var_handler.HEIGHT_L4, global_var_handler.WIDTH_L4, 3)
        global_var_handler.L4_SIZE = (global_var_handler.HEIGHT_L4, global_var_handler.WIDTH_L4)
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L4_SIZE)

        global_var_handler.HEIGHT_L5 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 5))
        global_var_handler.WIDTH_L5 = int(math.ceil(global_var_handler.WIDTH_L0 >> 5))
        global_var_handler.L5_SIZE_RGB = (global_var_handler.HEIGHT_L5, global_var_handler.WIDTH_L5, 3)
        global_var_handler.L5_SIZE = (global_var_handler.HEIGHT_L5, global_var_handler.WIDTH_L5)
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L5_SIZE)
        
        global_var_handler.HEIGHT_L6 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 6))
        global_var_handler.WIDTH_L6 = int(math.ceil(global_var_handler.WIDTH_L0 >> 6))
        global_var_handler.L6_SIZE_RGB = (global_var_handler.HEIGHT_L6, global_var_handler.WIDTH_L6, 3)
        global_var_handler.L6_SIZE = (global_var_handler.HEIGHT_L6, global_var_handler.WIDTH_L6)
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L6_SIZE)

        global_var_handler.HEIGHT_L7 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 7))
        global_var_handler.WIDTH_L7 = int(math.ceil(global_var_handler.WIDTH_L0 >> 7))
        global_var_handler.L7_SIZE_RGB = (global_var_handler.HEIGHT_L7, global_var_handler.WIDTH_L7, 3)
        global_var_handler.L7_SIZE = (global_var_handler.HEIGHT_L7, global_var_handler.WIDTH_L7)      
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L7_SIZE)

        global_var_handler.HEIGHT_L8 = int(math.ceil(global_var_handler.HEIGHT_L0 >> 8))
        global_var_handler.WIDTH_L8 = int(math.ceil(global_var_handler.WIDTH_L0 >> 8))
        global_var_handler.L8_SIZE_RGB = (global_var_handler.HEIGHT_L8, global_var_handler.WIDTH_L8, 3)
        global_var_handler.L8_SIZE = (global_var_handler.HEIGHT_L8, global_var_handler.WIDTH_L8)
        global_var_handler.SIZE_ARRAY.append(global_var_handler.L8_SIZE)
        
        global_var_handler.STR_L0_SIZE = str(global_var_handler.HEIGHT_L0) + 'x' + str(global_var_handler.WIDTH_L0)


    @classmethod
    def get_size_equivalence(cls, level):
        """
        :param cls:
        :param level: string with pyramid level
        :return: Returns the attribute from the class equivalent to string
        """
        try:
            # backward compatibility
            if 'SIZE' in level:
                return eval('global_var_handler.' + level)
            else:
                return eval(level)
        except BaseException as error:
            log_error_to_console("SIZE IS NOT A PYRAMID LVL: ", str(error))
            return global_var_handler.L0_SIZE
