from Utils.log_handler import log_to_console
from Application.Frame.port import Port

"""
Module handles the manipulation of transfer ports for the APPL block
"""

# Dictionaries to hold the transfer ports for each waves
portsDict = []
NR_WAVES = 0
ACTIVE_WAVE = 0


def create_ports_dict(nr_waves: int) -> None:
    """
    Creates the transfer ports list for a wave
    :return: None
    """
    global portsDict, NR_WAVES
    NR_WAVES = nr_waves

    for wave in range(nr_waves + 1):
        portsDict.append(dict())


def add_port(name: str, size: int, port_type: str, is_image: bool, wave: int = 0) -> None:
    """
    Add a port in the port dictionary.
    The port is automatically added to all slots for waves
    :param name: name of the port
    :param size: size of the port
    :param port_type: type of the port
    :param is_image: if arr is an image
    :param wave: wave of the port
    :return: None
    """
    portsDict[wave][name] = Port(name=name, size=size, port_type=port_type, is_image=is_image)


def reshape_ports(size_array: list) -> None:
    """
    Add a port in the port dictionary.
    The port is automatically added to all slots for waves
    :param size_array: list of resolutions of levels.
    :return: None
    """
    for el in portsDict[ACTIVE_WAVE].keys():
        if 'LC' not in el:
            port_to_change = portsDict[ACTIVE_WAVE][el]
            if port_to_change.get_is_image() is True:
                channels = len(port_to_change.arr.shape)
                level_to_change = int(port_to_change.name[-1])
                if channels == 2:
                    port_to_change.reshape_arr(size_new_array=(size_array[level_to_change][0], size_array[level_to_change][1]),
                                               type_new_array=port_to_change.arr.dtype)
                if channels == 3:
                    port_to_change.reshape_arr(size_new_array=(size_array[level_to_change][0], size_array[level_to_change][1], channels),
                                               type_new_array=port_to_change.arr.dtype)


def exist_port(name: str) -> bool:
    """
    Check if ports exists.
    :param name: port name
    :return: if ports exists
    """
    return name in portsDict[ACTIVE_WAVE].keys()


def get_port_from_wave(name: str, wave_offset: int = 0) -> Port:
    """
    Get port from a specific wave
    :param name: name of the port
    :param wave_offset: offset to current wave
    :return: Corresponding port of current wave
    """
    global NR_WAVES
    return portsDict[(ACTIVE_WAVE - wave_offset) % NR_WAVES].get(name)


def prepare_ports_new_wave(frame: int) -> None:
    """
    Make internal mechanism for changing slots to work on new wave
    :return: None
    """
    global ACTIVE_WAVE, NR_WAVES

    ACTIVE_WAVE = frame % NR_WAVES

    for el in portsDict[ACTIVE_WAVE].keys():
        port_to_change = portsDict[ACTIVE_WAVE][el]
        port_to_change.self_reset()

def set_invalid_ports_of_job(ports: list) -> None:
    """
    Set's invalid all the ports in the list
    :param ports: list of ports
    :return: None
    """
    for port in ports:
        portsDict[ACTIVE_WAVE][port[0]].reset_valid_flag()


def debug_ports_job(port_type: str, ports: list) -> None:
    """
    Debug information for ports.
    :param port_type: Type of port
    :param ports: port
    """
    log_to_console(port_type)
    # print(ports)
    if port_type == 'input':
        for port in ports:
            log_to_console("PORT: {port:150s} is in STATE: {state}".format(port=portsDict[ACTIVE_WAVE][port].name,
                                                                           state=portsDict[ACTIVE_WAVE][port].isValid))
    else:
        for port in ports:
            log_to_console("PORT: {port:150s} is in STATE: {state}".format(port=portsDict[ACTIVE_WAVE][port[0]].name,
                                                                           state=portsDict[ACTIVE_WAVE][port[0]].isValid))


def log_to_console_exchange_ports() -> None:
    """
    Logs to console all the ports created
    :return:None
    """
    global NR_WAVES
    log_to_console("Exchange ports created:")
    for wave in range(NR_WAVES):
        log_to_console('WAVE: {wave}'.format(wave=str(wave)))
        for port, memory in portsDict[wave].items():
            log_to_console('PORT: {port:150} at ADDRESS: {size:100}'.format(port=str(port), size=str(memory)))


if __name__ == "__main__":
    pass
