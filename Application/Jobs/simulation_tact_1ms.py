import time

# help variables
VAR = 10000
CONVERT_TO_MS = 1000
ERROR_ACCEPTED = 0.0000000000001


# noinspection PyUnusedLocal
def adjust_var() -> None:
    """
    Function to help adapt this test module the HW
    :return: None
    """
    one_ms = 1.0000
    end_time = 99999
    initial_time = 0

    while end_time - one_ms > ERROR_ACCEPTED:
        print("start")
        global VAR

        x = VAR
        print(x)
        initial_time = time.time()
        while x:
            x -= 1
        end_time = (time.time() - initial_time) * CONVERT_TO_MS

        if end_time > one_ms:
            VAR -= 1
        else:
            VAR += 1
        print(end_time)


def simulate_job(ms: float) -> None:
    """
    Function holds system occupied for ms time
    :param ms: number of ms to hold the system.
    :return:
    """
    end_time = 0
    x = 0
    initial_time = time.time()

    while end_time - ms < 0.0:
        while x:
            x += 1
        end_time = (time.time() - initial_time) * CONVERT_TO_MS


if __name__ == "__main__":
    pass
