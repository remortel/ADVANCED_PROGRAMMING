import asyncio
import time

async def main():
    print('Tim')
    # await foo('text') # this implies that we have to wait for foo to be finished
    task = asyncio.create_task(foo('text2'))
    # await task # this implies you have to wait for task to be finished
    await asyncio.sleep(0.5) # try changing the sleep time from 2 to 0.5 seconds
    print('finished')
    await asyncio.sleep(2.5) # try changing the sleep time from 2 to 0.5 seconds
async def foo(text):
    # defining a coroutine via await statement
    await asyncio.sleep(2.5)
    print(text)


start_time = time.perf_counter()
asyncio.run(main())
stop_time = time.perf_counter()
execution_time = (stop_time - start_time)
print(f"The code took {execution_time} seconds to run.")
