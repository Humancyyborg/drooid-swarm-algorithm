import logging
import os


def log_info(label_name='left', folder_name='pos', file_name='pos_v1', logger_name='PositionLogger', data=None):
    # Make folder given label name
    label_loc = os.path.join('../data', label_name)
    os.makedirs(label_loc, exist_ok=True)
    # Make folder given folder name
    folder_loc = os.path.join(label_loc, folder_name)
    os.makedirs(folder_loc, exist_ok=True)

    # Create folders to store log files if they don't exist
    log_file = os.path.join(folder_loc, file_name+'.txt')

    # Configure logging for position
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)

    # Log the sample data
    num_elements = {
        'thrusts': 4,
        'obs': 18,
    }.get(folder_name, 3)

    # Log the elements
    for item in data:
        logger.info(f'[{", ".join(map(str, item[:num_elements]))}]')

    # for item in data:
    #     if folder_name == 'thrusts':
    #         logger.info(f'[{item[0]}, {item[1]}, {item[2]}, {item[3]}]')
    #     elif folder_name == 'obs':
    #         logger.info(f'[{item[0]}, {item[1]}, {item[2]}, {item[3]}, {item[4]}]')
    #     else:
    #         logger.info(f'[{item[0]}, {item[1]}, {item[2]}]')

    # Close the logging handlers to release resources and flush data to disk
    handler.close()
    # Remove handlers from the logger objects
    logger.handlers = []
