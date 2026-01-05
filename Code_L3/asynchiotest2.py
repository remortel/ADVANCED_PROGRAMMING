import asyncio
import time

async def fetch_data():
    print('start fetching')
    await asyncio.sleep(2.5)
    print('done fetching')
    return {'data': 1}
async def print_numbers():
    for i in range(10):
        print(i)
        # tray changing the sleep time from 0.25 to 0.5 seconds
        await asyncio.sleep(0.5)
async def main():
    # a task that returns a value creates a future
    task1=asyncio.create_task(fetch_data())
    task2=asyncio.create_task(print_numbers())
    # try using this to see what happens when you don't await a task
    # value = task1
    value = await task1
    print(value)
    # You will see that task2 will not be completed, unless you await it
    await task2
    


start_time = time.perf_counter()
asyncio.run(main())
stop_time = time.perf_counter()
execution_time = (stop_time - start_time)
print(f"The code took {execution_time} seconds to run.")
