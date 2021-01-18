from Application.Utils.TimeLogger import Timer
from Application.Frame.transferJobPorts import add_port, exist_port, set_invalid_ports_of_job
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

"""
Module handles the jobs for the APPL block
"""


class JobState:
    """
    Class for job states enumeration
    """
    # the job is NOT_INIT state not yet activated or inited
    NOT_INIT = "NOT_INIT"
    # the job is in INIT state, this state should happen once per APPL run
    INIT = "INIT"
    # the job is in RUN state, state where the main activity of this is done
    RUN = "RUN"
    # the job is in OFF state
    TERMINATE = "OFF"


class Job:
    """
    Class that describes the APPL job
    """

    # initial state of a job
    __state__ = JobState.NOT_INIT

    def __init__(self, name: str, main_function, init_function=None, output_ports: list = None, input_ports: list = None,
                 init_func_param: list = None, main_func_param: list = None, waves: int = 1) -> None:
        """
        Constructor of job class
        At creation the job state will be NOT_INIT
        :param name: name of job
        :param main_function: address of main function
        :param init_function: address of init function
        :param output_ports: list of ports(data) that the job will give out
        :param input_ports: list of ports(data) are necessary for job to run
        :param init_func_param: parameters for init function
        :param main_func_param: parameters for main function
        """

        # create list of ports
        if output_ports is None:
            output_ports = []

        if input_ports is None:
            input_ports = []

        # initiate timer
        self.__timer__ = Timer()
        # set starting state to NOT_INIT
        __state__ = JobState.NOT_INIT

        self.__name__ = name
        self.__init_function__ = init_function
        self.__init_func_param__ = init_func_param
        self.__main_function__ = main_function
        self.__main_func_param__ = main_func_param
        self.__input_ports__ = input_ports
        self.__output_ports__ = output_ports

        # get output ports
        for wave in range(waves):
            for pOut in output_ports:
                add_port(name=pOut[0], size=pOut[1], port_type=pOut[2], is_image=pOut[3], wave=wave)

        log_to_console('JOB : {job:150s} is CREATED.'.format(job=self.__name__))

    def init(self) -> None:
        """
        Runs the init function for the job
        If init function of job runs OK the state of the jobs changes to INIT
        :return: None
        """
        if self.verify_input_ports():
            if self.__init_function__ is not None:
                if self.__init_func_param__ is None:
                    self.__state__ = self.__init_function__()
                else:
                    self.__state__ = self.__init_function__(self.__init_func_param__)
            else:
                self.__state__ = JobState.INIT
            log_to_file('{} Avg Time[ms]'.format(self.__name__))
        else:
            log_error_to_console("JOB : {} input ports not valid!".format(self.__name__))

        if self.__state__ is not JobState.NOT_INIT:
            log_to_console(self.get_echo())

    def run(self) -> None:
        """
        Runs the main function for the job
        For this function to run the state of the job should be INIT.
        If the job doesn't run OK the job will be dropped
        :return: None
        """
        if self.__state__ != JobState.NOT_INIT:
            if self.__state__ != JobState.TERMINATE:
                self.__timer__.start_cycle_timer()
                self.__state__ = JobState.RUN

                # set invalid flag to ports to be sure that nobody uses invalidated data
                set_invalid_ports_of_job(ports=self.__output_ports__)

                # Runs the main function of job
                if self.__main_function__(self.__main_func_param__) is False:
                    log_to_console('ERROR: JOB {job:150s} DROPPED. INPUT NOK!'.format(job=self.__name__))
                else:
                    log_to_console(str(self.get_echo()))

                self.__timer__.end_cycle_timer()
                self.__timer__.cycle_updater()
            else:
                log_to_console("JOB : {job:150s} IS TERMINATED!".format(job=self.__name__))
        else:
            log_to_console("JOB : {job:150s} IS NOT INITED!".format(job=self.__name__))

    def verify_input_ports(self) -> bool:
        """
        Check if all the input ports all OK
        :return: True if all ports are valid
        """
        for p in self.__input_ports__:
            if exist_port(name=p) is not True:
                log_error_to_console("{} input port {} is not inited!".format(self.__name__, p), '')
                return False
        return True

    def terminate(self) -> None:
        """
        Terminates job.
        Set's the state to TERMINATE
        :return: None
        """
        self.__state__ = JobState.TERMINATE
        log_to_console(self.get_echo())

    def get_echo(self) -> str:
        """
        :return: string containing the NAME JOB is in state STATE
        """
        return 'JOB : {job:150s} is in STATE: {state:50s}'.format(job=self.__name__, state=self.__state__)

    def get_time(self) -> float:
        """
        :return: current time
        """
        return self.__timer__.get_current_time()

    def get_average_time(self) -> float:
        """
        :return: average time of job
        """
        return self.__timer__.get_average_time()

    def get_out_ports(self):
        """
        :return: output port list of job
        """
        return self.__output_ports__
