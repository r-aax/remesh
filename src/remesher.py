import sys
import logging
from logging import StreamHandler, Formatter

# Log.
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
log.addHandler(handler)


class Remesher:
    """
    Main remesher class.
    """

    def __init__(self):
        """
        Constructor.
        """

        # Time for remesh.
        self.remesh_time = 0.0

        # Log.
        self.log = log
