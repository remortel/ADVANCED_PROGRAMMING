from multiprocessing import Process, Queue
import os
import time

def square(numbers, queue):
    for i in numbers:
        time.sleep(0.5)
        queue.put(i*i)
def cube(numbers, queue):
    for i in numbers:
        time.sleep(0.5)
        queue.put(i*i*i)

if __name__ == "__main__":

    numbers = range(50)
    queue = Queue()

    square_process = Process(target=square, args=(numbers,queue))
    cube_process = Process(target=cube, args=(numbers, queue))

    print('start square')
    square_process.start()
    cube_process.start()

    square_process.join()
    cube_process.join()
    print('processes finished', queue.empty() )
    
    while not queue.empty():
        print(queue.get())
    print("All Done!\n")
