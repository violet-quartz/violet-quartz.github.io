---
title: '对 asyncio 更详细的介绍'
description: '本文介绍了 asyncio 的基本使用以及背后的事件循环和调度机制，总结了协程之间的同步机制，并介绍了协程的常用场景，对于计算密集型的 python 任务可以通过 loop.run_in_executor 丢到进程池中'
pubDate: '2026-09-08'
tags: ['asyncio', 'python', 'parallel programming', 'concurrency']
---

# asyncio

asyncio 是 Python 用于编写并发代码的标准库，基于协程（coroutine）实现单线程内的异步 I/O。关于 Python 的并行与并发编程的背景知识，可以参见博客[《Python Parallel and Concurrent Programming》](/blog/python-parallel-and-concurrent-programming/)，这里主要对 asyncio 做一个简单介绍。


## 1 基础概念和使用

启动异步程序的入口: `asyncio.run()`

定义一个协程：用 `async def` 定义一个函数

### 1.1 协程的执行

首先需要区分协程对象、Future、 Task 这三个概念

| 概念 | 说明 |
|---|---|
| **协程对象（coroutine）** | `async def` 函数调用后返回的对象，本身**不会**被调度执行，只是一个"计划" |
| **Future** | 表示"未来某个时刻会有结果"的占位符，事件循环通过它来跟踪异步操作状态 |
| **Task** | `Future` 的子类，专门用来**包装协程**并交给事件循环调度执行 |

```python
sync def worker(n):
    print(f"worker {n} 开始")
    await asyncio.sleep(1)
    print(f"worker {n} 结束")

async def main():
    coro1 = worker(1) # 只是创建了协程对象，还没执行
    coro2 = worker(2) # 只是创建了协程对象，还没执行

    # 裸的协程对象必须被 await 或包装成 Task 才会执行
    await coro1 # 执行 coro1，并等到 coro1 完成后才继续执行
    task2 = asyncio.create_task(coro2) # 将 coro2 包装成 Task 并调度执行

asyncio.run(main())

# 输出为
# worker 1 开始
# worker 1 结束
# worker 2 开始
```

只有 Task（或 Future）才会被事件循环真正"盯着跑"，裸的协程对象必须被 await 或包装成 Task 才会执行。 await 能等到 task 执行完成。

### 1.2 原理：事件循环和调度机制

事件循环（Event Loop）是 asyncio 的核心引擎，本质上是一个无限循环，运行在单线程里，靠"在等待时切换到别的任务"来实现并发，并不是真正的并行。

```python
loop = asyncio.get_running_loop()
```

每一轮循环：
1) 计算最近的定时任务还要多久到期 
2) 用 select/epoll 等待 I/O 事件或超时
3) 把到期的定时任务和就绪的 I/O 回调移入就绪队列
4) 依次执行就绪队列里的所有回调    
5) 回到第1步，循环往复        

asyncio 是协作式多任务（cooperative multitasking），不是像操作系统线程那样的抢占式调度。这意味着：
- 一个协程会一直独占 CPU，直到它主动让出控制权（通过 await）
- 事件循环不会强行打断一个正在运行的协程

调度流程总结：

```
asyncio.run(main())
        │
        ▼
  创建事件循环 loop
        │
        ▼
  把 main() 包装成 Task，加入调度队列
        │
        ▼
  loop.run_forever() 开始循环
        │
        ├─► 执行就绪任务，遇到 await 就暂停并登记"何时/何事件唤醒"
        ├─► 检查定时器堆，到期的移入就绪队列
        ├─► 用 select/epoll 检查 I/O，就绪的移入就绪队列
        └─► 循环直到所有 Task 完成 → loop.stop()
```


create_task() 会立即把协程包装成 Task 并加入事件循环的调度队列（但不会立刻运行，要等当前协程让出控制权）

```python
import asyncio

async def worker(n):
    print(f"worker {n} 开始")
    await asyncio.sleep(1)
    print(f"worker {n} 结束")

async def main():
    await asyncio.gather(worker(1), worker(2))
    t3 = asyncio.create_task(worker(3)) # create_task() 会立即把协程包装成 Task 并加入事件循环的调度队列
    t4 = asyncio.create_task(worker(4))
    await asyncio.gather(t3, t4) 
    t5 = asyncio.create_task(worker(5))
    t6 = asyncio.create_task(worker(6)) 

asyncio.run(main())
```

上面的运行结果：
```
worker 1 开始
worker 2 开始
worker 1 结束
worker 2 结束
worker 3 开始
worker 4 开始
worker 3 结束
worker 4 结束
worker 5 开始
worker 6 开始
```

`await asyncio.gather()` 等到所有并发结束的结果。


## 2 协程间的同步机制

| 原语 | 适用场景 |
|---|---|
| `asyncio.Event` | 一对多广播通知（"发生了某事"） |
| `asyncio.Lock` | 保护共享资源，同一时刻只允许一个协程访问 |
| `asyncio.Condition` | 更复杂的等待/通知逻辑（可结合谓词判断） |
| `asyncio.Queue` | 协程间传递数据，一对一或多对多的生产消费模型 |
| `asyncio.Semaphore` | 限制同时访问某资源的协程数量 |

## 3 运用

**asyncio 的并发本质是：单线程 + 协作式调度，靠协程在 `await` 处主动让出控制权，事件循环负责在"定时器到期"和"I/O 就绪"时把挂起的协程重新唤醒并放回执行队列**——没有真正的并行，但对 I/O 密集型任务而言，"该等的时候不浪费 CPU 去等，转去做别的事"就已经能带来巨大的效率提升。

对于 CPU 密集型任务，由于 Python 的 GIL 特性，可以通过 loop.run_in_executor 丢到进程池。

对于 "阻塞但不太耗 CPU"的任务，可以通过 loop.run_in_executor 丢到线程池中。

可以参考代码 [cpu_bound_vs_blocking_fetch.py](https://github.com/violet-quartz/code-snippets/blob/main/asyncio/cpu_bound_vs_blocking_fetch.py)

| 场景 | 推荐 | 原因 |
|---|---|---|
| 调用同步的阻塞 I/O 库（如某些数据库驱动、`requests`） | 线程池 | 线程在等待 I/O 时会释放 GIL，其他线程/主循环可以运行 |
| 纯 CPU 密集计算（哈希、图像处理、数值运算） | 进程池 | 绕开 GIL，真正并行，多核加速 |
| 少量、轻量的同步调用 | 默认线程池 (`None`) | 简单，无需额外管理 |
| 大量重计算任务，需要压榨多核性能 | 自定义 `ProcessPoolExecutor`，并控制进程数 | 避免频繁创建线程/进程的开销 |


