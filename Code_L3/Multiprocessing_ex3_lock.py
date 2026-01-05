from multiprocessing import Process, Lock, Value
from multiprocessing import log_to_stderr, get_logger
import time
import logging


def add_500_lock(total,lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire()
        total.value +=5
        lock.release()

def sub_500_lock(total,lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire()
        total.value -=5
        lock.release()


if __name__ == "__main__":

    total = Value('i',500)
    lock = Lock()

    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    add_process = Process(target=add_500_lock, args=(total,lock))
    sub_process = Process(target=sub_500_lock, args=(total,lock))

    add_process.start()
    sub_process.start()

    add_process.join()
    sub_process.join()

    print(total.value)
    print("All Done!\n")
