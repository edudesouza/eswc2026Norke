
import time
from rich import print

def diff_time(legenda,inicio):

    fim = time.time()
    tpo  = fim - inicio

    print( f'[blue]{legenda}{tpo:.2f}s\n' )

    return
