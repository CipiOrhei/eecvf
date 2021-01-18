from Utils.log_handler import log_to_file, log_end_of_wave

"""
Module handles the Round Robin scheduler used by APPL layer for execution
"""


def run_rr(jobs: list) -> None:
    """
    Simple round robin scheduler
    :param jobs: list of Job objects
    :return: None
    """
    for job in jobs:
        job.run()
        log_to_file(str(job.get_time()))
    log_end_of_wave()


if __name__ == "__main__":
    pass
