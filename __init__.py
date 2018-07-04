import sys
from os import path, remove
import logging
import logging.config

current_path = path.dirname(path.abspath(__file__))
sys.path.append(current_path)

# If applicable, delete the existing log file to generate a fresh log
#  file during each execution
if path.isfile(current_path + "/temp/altcoin.log"):
    remove(current_path + "/temp/altcoin.log")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=current_path + '/temp/altcoin.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger(__name__).addHandler(console)
