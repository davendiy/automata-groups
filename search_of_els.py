#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 28.02.2020
# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

from multiprocessing import Value, Process
from src.source2 import *
import time

done1 = Value('f', 0)
done2 = Value('f', 0)
done3 = Value('f', 0)

memory1 = Value('i', 0)
memory2 = Value('i', 0)
memory3 = Value('i', 0)


def info():
    while True:
        print('\r', end='')
        memory = memory1.value + memory2.value + memory3.value
        print(f"[*] done1: {str(done1.value)[:5]}%;  "
              f"done2: {str(done2.value)[:5]}%; "
              f"done3: {str(done3.value)[:5]}%; "
              f"memory usage: {memory / (CHUNK * 3) * 100}%;     ", end='')
        time.sleep(1)

        if done1.value >= 99.999 and done2.value >= 99.999 \
                and done3.value >= 99.999:
            break


def task(n, done, memory):
    process = psutil.Process(os.getpid())

    all_amount = 3 * 2 ** (n - 1)
    res = []

    i = 0
    try:
        print(f"[!!!] Start the process for n={n}")
        for el in permute('abc', repeat=n):

            tmp_el = from_string(el)
            if tmp_el.permutation != TRIVIAL_PERM and all([child.is_trivial() for child in tmp_el]):
                print(tmp_el)
                res.append(tmp_el)
            i += 1

            if memory.value > CHUNK:
                AutomataGroupElement.clear_memory()
                initial_state()

            with done.get_lock():
                done.value = i / all_amount * 100
            with memory.get_lock():
                memory.value = process.memory_info()[0]

    except Exception as exc:
        print("[ERROR]", exc.with_traceback())
        
    finally:
        with open(f'seach_results_n{n}.txt', 'w') as file:
            file.write(str(res))


Process(target=task, args=(20, done1, memory1), daemon=True).start()
Process(target=task, args=(21, done2, memory2), daemon=True).start()
Process(target=task, args=(22, done3, memory3), daemon=True).start()

obs = Process(target=info)
obs.start()
obs.join()
