---
layout: post
title:  "Python Parallel and Concurrent Programming"
date:   2023-12-22 10:00 +0800
categories: note
subtype: python
istop: true
---
Parallel and concurrent programming are powerful tools to increase processing throughput and to write 
faster and efficient applications. Here we'll summarize the main concepts of parallel and concurrent 
programming and show how it works in Python.

# Table of Contents
- [Threads and Processes](#threads-and-processes)
    - [Thread vs. Process](#thread-vs-process)
    - [Concurrency vs. Parallelism](#concurrency-vs-parallelism)
    - [Python GIL](#python-gil)
    - [IO-bound task vs. CPU-bound task](#io-bound-task-vs-cpu-bound-task)
    - [Thread](#thread)
    - [Process](#process)
- [Synchronization](#synchronization)
    - [Locks](#locks)
        - [Primitive lock](#primitive-lock)
        - [Try lock:Non-blocking acquire](#try-lock-non-blocking-acquire)
        - [Reentrant lock](#reentrant-lock)
    - [Condition variables](#condition-variables)
    - [Semaphores](#semaphores)
    - [Event Objects](#event-objects)
    - [Timer Objects](#timer-objects)
    - [Barriers](#barriers)
- [Asynchronous Tasks](#asynchronous-tasks)
    - [Future](#future)
    - [Thread Pool](#thread-pool)
    - [Process Pool](#process-pool)
- [External Libraries/Frameworks](#external-librariesframeworks)

## Threads and Processes
### Thread vs. Process
A process is an executing program. Processes are isolated. Each process has its own Process Control Block, Stack, and Address Space.

A thread is the unit of execution within a process. A process has at least one thread. It's called the main thread. A process can have multiple threads. Each thread has its own stack, registers and program counters. Threads within a process share a memory address space. It's possible to communicate between threads using that shared memory space. However, one misbehaving thread could bring down the entire process. 

A thread is the basic unit to which the operating system allocates processor time. Context switching of process uses an interface in an operating system. Context switching of thread does not require calling an operating system and is more light weight.

### Concurrency vs. Parallelism
Concurrency is when two or more tasks can start, run, and complete in overlapping time periods. It doesn't necessarily mean they'll ever both be running at the same instant. For example, multitasking on a single-core machine.

Parallelism is when tasks literally run at the same time, e.g. on a multicore processor.

### Python GIL
In CPython, the global interpreter lock, or GIL, is a mutex that protects access to Python objects, allowing only one thread to execute Python bytecodes. The GIL prevents race conditions and ensures thread safety. The mutex is necessary mainly because CPython's memory managment is not thread-safe.

To make CPython programs to take full advantage of multiprocessor systems, one can:
- Implement blocking or long-running operations in C/C++ outside the GIL, such as image processing, and Numpy number curncing.
- Use multiprocessing module.

Non-CPython implementations:
- [Jyhton](https://wiki.python.org/moin/Jython) and [IronPython](https://wiki.python.org/moin/IronPython) have no GIL and can fully exploit multiprocessor systems.
- [PyPy](https://wiki.python.org/moin/PyPy) currently has a GIL like Cpython.
- in Cython the GIL exists, but can be released temporarily using a "with" statement.

### IO-bound task vs. CPU-bound task
IO-bound task means its progress rate is limited by the speed of the I/O system, such as task associated with disks or networking communication. CPU-bound task means its progress rate is limited by the speed of CPU, such as computation task.

CPU-bound tasks only really gain from using multiprocessing, while for IO-bound tasks, multithreading/concurrent programming helps.

### Thread
```python
import threading
import time

def func(message, **kwargs):
    name = threading.current_thread().name
    id = threading.current_thread().native_id
    print(f'thread id={id}, name={name}, message={message}')
    for k, w in kwargs.items():
        print(f'{name}: {k}={w}')
        time.sleep(0.1)

threading.Thread(target=func, name='Jane', 
                 args=('hello',), kwargs={'gender':'female', 'age': 34}).start()
threading.Thread(target=func, name='Zach',
                 args=('hi',), kwargs={'gender':'male', 'age':60, 'occupation':'Teacher'}).start()    
```
Output:
```
thread id=21660, name=Jane, message=hello
Jane: gender=female
thread id=6152, name=Zach, message=hi
Zach: gender=male
Jane: age=34
Zach: age=60
Zach: occupation=Teacher
```
Daemon(Background) thread does not prevent the process from terminating. By default, the daemonic property is inherited from the creating thread. The main thread is not a daemon thread and therefore all threads created in the main thread default to non-daemon thread.
```python
import threading
import time

def func(message, **kwargs):
    name = threading.current_thread().name
    id = threading.current_thread().native_id
    print(f'thread id={id}, name={name}, message={message}')
    for k, w in kwargs.items():
        print(f'{name}: {k}={w}')
        time.sleep(0.1)

   
t2 = threading.Thread(target=func, name='Zach',
                args=('hi',), kwargs={'gender':'male', 'age':60, 'occupation':'Teacher'})
t2.daemon = True
t2.start()
threading.Thread(target=func, name='Jane', 
                args=('hello',), kwargs={'gender':'female', 'age': 34}, daemon=True).start()
```
Output:
```
thread id=15568, name=Zach, message=hi
Zach: gender=male
thread id=24620, name=Jane, message=hello
Jane: gender=female
```
### Process
```Python
import multiprocessing as mp
import random
import math

def compute_sum(arr, start_idx, end_idx, q):
    sum = 0
    for i in range(start_idx, end_idx):
        sum += arr[i]
    q.put(sum) 

print('Hi! My name is', __name__)
if __name__ == '__main__': # This line is necessary.
    arr = [random.random() for _ in range(1_000_000)]
    q = mp.Queue()
    blocks = 10
    block_size = math.ceil(len(arr) / blocks)
    workers = []
    for i in range(blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, len(arr))
        workers.append(mp.Process(target=compute_sum(arr, start_idx, end_idx, q)))
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    seq_sum = sum(arr)
    par_sum = 0
    while not q.empty():
        t = q.get()
        par_sum += t
    assert abs(seq_sum - par_sum) < 0.0001
```
Output:
```
Hi! My name is __main__
Hi! My name is __mp_main__
Hi! My name is __mp_main__
Hi! My name is __mp_main__
...
```
## Synchronization
Here we use thread-based parallelism as example.
### Locks
#### Primitive lock

A primitive lock is in one of two states, "locked" or "unlocked". It's currently the lowest level synchronization primitive available. It can be released by different threads than the one acquire it(but not recommended).

```python
import threading
some_lock = threading.Lock()

some_lock.acquire()
# do something...
some_lock.release()
```
Remember to release the lock if there is exception thrown.
```python
some_lock.acquire()
try:
    # do something...
finally:
    some_lock.release()
```
which is equivalent to:
```python
with some_lock:
    # do something...
```
#### Try lock: Non-blocking acquire

Non-blocking lock.acquire method for mutex. If the mutex is available, lock it and return TRUE. If the mutex is not available, immediately return FALSE.

```python
import threading
some_lock = threading.Lock()

if some_lock.acquire(blocking=False):
    # do something else

if some_lock.acquire(blocking=True, timeout=1):
    # block for at most 1 second
```

#### Reentrant lock

A reentrant lock is a synchronization primitive that may be acquired multiple times by the same thread. Internally, it uses the concepts of "owning thread" and "recursion level" in addition to the locked/unlocked state used by primitive locks. For reentrant lock, it can only be released by the thread which acquires it.

```python
import threading

red_cnt = 0
blue_cnt = 0
rlock = threading.RLock()

def add_red():
    global red_cnt
    with rlock:
        red_cnt += 1

def add_pair():
    global blue_cnt
    rlock.acquire()
    blue_cnt += 1
    add_red()
    rlock.release()

if __name__ == '__main__':
    ths = []
    for i in range(3):
        ths.append(threading.Thread(target=add_pair))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f'red_cnt={red_cnt}, blue_cnt={blue_cnt}')
```
### Condition variables
Spinning, which is busy waiting, repeatedly acquiring and releasing the lock to check for a certain condition to continue. To avoid this, we introduce condition variable.Condition variable, serves as a queue of threads waiting for a certain condition. The condition variable works with a mutex together to implement the higher level construct called a monitor.

A monitor, protects section of code with mutual exclusion and provides ability for threads to wait until a condition has become true along with a mechanism to signal those waiting threads when their condition has been met.

Conditional variable has three main operations: wait, signal/notify, broadcast/notifyall. And it support context management protocol.

```python
import threading
from collections import deque
import time

rlock = threading.RLock()
BufferNotFull = threading.Condition(lock=rlock)
BufferNotEmpty = threading.Condition(lock=rlock)

q = deque()
capcaity = 5
closed = False

def produce():
    global capacity
    name = threading.current_thread().name
    while not closed:
        with rlock:
            while len(q) >= capcaity:
                BufferNotFull.wait()
            print(f'{name} produces one item.')
            q.append('item')
            BufferNotEmpty.notify()

def consume():
    name = threading.current_thread().name
    while not closed:
        with rlock:
            while len(q) == 0:
                BufferNotEmpty.wait()
            q.popleft()
            print(f'{name} consumes one item.')
            BufferNotFull.notify()

if __name__ == '__main__':
    for i in range(2):
        threading.Thread(target=produce).start()
    for i in range(3):
        threading.Thread(target=consume).start()
    time.sleep(0.2)
    closed = True
```
Note: the notify() and notify_all() methods don't release the lock; this means that the thread or threads awakened will not return from their wait() call immediately, but only when the thread that called notify() or notify_all() finally relinquishes ownership of the lock.

### Semaphores
Semaphore is one of the oldest synchronization primitives. Semaphore can be used by multiple threads at the same time. It manages an internal counter which is decremented by each acquire() call and incremented by each release() call. When acquire() finds the counter is zero, it blocks, waiting until some other thread calls release(). Semaphores also support the context management protocol.

Semaphores are often used to gurard resources with limited capacity.

```python
import threading

max_connections = 5
semaphore = threading.Semaphore(value=max_connections)

with semaphore:
    conn = connectdb()
    try:
        # ... use connection ...
    finally:
        conn.close()
```
### Event Objects
Event is one of the simplest mechanisms for communication between threads: one thread signals an event and other threads wait for it.

```python
import threading
import time

elem_put = threading.Event()
elem_get = threading.Event()
q = []

def produce():
    for num in range(3):
        elem_get.wait()
        q.append(num)
        print(f'Produce {num}')
        elem_get.clear()
        elem_put.set()

def consume():
    while elem_put.wait(timeout=0.5):
        print(f'Consume {q[-1]}')
        elem_put.clear()
        elem_get.set()

if __name__ == '__main__':
    threading.Thread(target=produce).start()
    threading.Thread(target=consume).start()
    time.sleep(0.2)
    elem_get.set()
```
Output:
```
Produce 0
Consume 0
Produce 1
Consume 1
Produce 2
Consume 2
```
### Timer Objects
Timer class represents an action that should be run only after a certian amount of time has passed. Timer is a subclass of Thread.

```python
import threading

def hello():
    print('hello, thanks for waiting.')

t = threading.Timer(interval=3, function=hello)
t.start()
# t.cancel() # Stop the timer and cancel the execution 
# of the timer's action. This will only work if the timer
# is still in its waiting stage.
```

### Barriers
Barrier is a synchronization primitive which prevents a group of threads from proceeding until enough threads have reached the barrier. Each of the threads tries to pass the barrier by calling the wait() method and will block until all of the threads have made their wait() calls. At this point, the threads are released simultaneously.

The barrier can be reused any number of times for the same number of threads.

```python
import threading

b = threading.Barrier(2, timeout=5)

def start_server():
    # do something ...
    print('Server started')

def server():
    start_server()
    b.wait()
    while True:
        connection = accept_connection()
        process_server_connection(connection)

def client():
    b.wait()
    while True:
        connection = make_connection()
        process_client_connection(connection)
```

## Asynchronous Tasks
The concurrent.futures module provides a high-level interface for asynchronously executing callables. The asynchronous execution can be performed with threads, using ThreadPoolExecutor, or separate processes, using ProcessPoolExecutor. Both implement the same interface and support context management protocol.

Use thread/process pool can reduce the complexity of management and the overhead in thread/process creation and destruction.
### Future
The concurrent.futures.Future class encapsulates the asynchronous execution of a callable. It's a placeholder for a result that will be available later and a mechanism to access the result of an asynchronous operation. Future instances are created by Executor.submit().
### Thread Pool

```python
import concurrent.futures 
import threading

def my_pow(num):
    name = threading.current_thread().name
    print(f'{name} compute pow(2, {num})')
    return pow(2, num)

if __name__ == '__main__':
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(my_pow, i) for i in range(6)]
    for f in concurrent.futures.as_completed(futures):
        print(f.result())
    pool.shutdown() 
```
Output:
```
ThreadPoolExecutor-0_0 compute pow(2, 0)
ThreadPoolExecutor-0_0 compute pow(2, 1)
2
1
ThreadPoolExecutor-0_1 compute pow(2, 3)
ThreadPoolExecutor-0_1 compute pow(2, 4)
ThreadPoolExecutor-0_1 compute pow(2, 5)
ThreadPoolExecutor-0_0 compute pow(2, 2)
8
16
32
4
```
### Process Pool

```python
import concurrent.futures 
import os

def my_pow(num):
    name = os.getpid()
    print(f'Process {name} compute pow(2, {num})')
    return pow(2, num)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(my_pow, i) for i in range(10, 16)]
        for f in concurrent.futures.as_completed(futures):
            print(f.result())
```
Output:
```
Process 4304 compute pow(2, 10)
Process 19172 compute pow(2, 11)
Process 4304 compute pow(2, 12)
1024
2048
Process 19172 compute pow(2, 13)
Process 19172 compute pow(2, 15)
Process 4304 compute pow(2, 14)
4096
8192
32768
16384
```


## External Libraries/Frameworks
- [asyncio](https://docs.python.org/3/library/asyncio.html): a library to write concurrent code using the async/await syntax.
- Celery: Task queues to distribute work across threads or machines.
- Pyro5: Python remote objects that can talk to each other over a network.
- RPyC: Remote procedure calls for distributed computing.
- mpi4py: Python wrapper for Message Passing Interface (MPI).
- PyCUDA: Python wrapper for the NVIDIA CUDA API.
