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
### Locks
1. Primitive lock
A primitive lock is in one of two states, "locked" or "unlocked". It's currently the lowest level synchronization primitive available. It can be released by different threads than the one acquire it(but not recommended).
```Python
import threading
some_lock = threading.Lock()

some_lock.acquire()
# do something...
some_lock.release()
```
Rember to release the lock if there is exception thrown.
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
2. Try lock: Non-blocking acquire
Non-blocking lock.acquire method for mutex. If the mutex is available, lock it and return TRUE. If the mutex is not available, immediately return FALSE.
```python
import threading
some_lock = threading.Lock()

if some_lock.acquire(blocking=False):
    # do something else

if some_lock.acquire(blocking=True, timeout=1):
    # block for at most 1 second
```
3. Reentrant lock
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
Condition variable, serves as a queue of threads waiting for a certain condition. The condition variable works with a mutex together to implement the higher level construct called a monitor.

A monitor, protects section of code with mutual exclusion and provides ability for threads to wait until a condition has become true along with a mechanism to signal those waiting threads when their condition has been met.

Conditional variable has three main operations: wait, signal/notify, broadcast/notifyall.
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

### Event Objects
### Timer Objects
### Barriers

## Asynchronous Tasks
### Thread Pool
### Process Pool
### Future

## External Libraries/Frameworks
- [asyncio](https://docs.python.org/3/library/asyncio.html): a library to write concurrent code using the async/await syntax.
- Celery: Task queues to distribute work across threads or machines.
- Pyro5: Python remote objects that can talk to each other over a network.
- RPyC: Remote procedure calls for distributed computing.
- mpi4py: Python wrapper for Message Passing Interface (MPI).
- PyCUDA: Python wrapper for the NVIDIA CUDA API.
