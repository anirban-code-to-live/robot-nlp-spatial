import json, sys
import logging, logging.config


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object
    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    # std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    std_out_format = '%(message)s'
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(console_handler)

    return logger