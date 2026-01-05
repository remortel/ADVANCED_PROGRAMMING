from multiprocessing import Pool
import os
import time

def sum_square(number):
    s = 0
    for i in range(number):
        s += i * i
    return s
def sum_square_with_mp(numbers):
    start_time = time.time()
    p = Pool()
    result = p.map(sum_square, numbers)

    p.close()
    p.join()

    end_time = time.time() - start_time
    nr_cores = os.cpu_count()
    print(f"Processing {len(numbers)} numbers took {end_time} time using {nr_cores} cores.")

def sum_square_no_mp(numbers):
    start_time = time.time()
    result = []
    for i in numbers:
        result.append(sum_square(i))
    end_time = time.time() - start_time

    print(f"Processing {len(numbers)} numbers took {end_time} time using one core.")

if __name__ == "__main__":

    numbers = range(100000) # try changing the range from 100 to 100000 to notice the gain
    sum_square_with_mp(numbers)
    sum_square_no_mp(numbers)

    print("All Done!\n")
