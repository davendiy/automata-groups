#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 13.02.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com


from src.source2 import *
from multiprocessing import Value, Process

import os
import psutil
import time

CHUNK = 1 * 1024 * 1024 * 1024

done1 = Value('f', 0)
done2 = Value('f', 0)
done3 = Value('f', 0)

memory1 = Value('i', 0)
memory2 = Value('i', 0)
memory3 = Value('i', 0)


def permute(seq, repeat):
    if repeat == 1:
        for el in seq:
            yield [el]
    elif repeat < 1:
        yield []
    else:
        for prev in permute(seq, repeat - 1):
            for el in seq:
                if prev[-1] == el:
                    continue
                yield prev + [el]


def info():
    while True:
        print('\r', end='')
        memory = memory1.value + memory2.value + memory3.value
        print(f"[*] done1: {str(done1.value)[:4]}%;  "
              f"done2: {str(done2.value)[:4]}%; "
              f"done3: {str(done3.value)[:4]}%; "
              f"memory usage: {memory / MEMORY_SIZE * 100}%;     ", end='')
        time.sleep(1)

        if done1.value >= 99.999 and done2.value >= 99.999 \
                and done3.value >= 99.999:
            break


def task(n, done, memory):
    _process = psutil.Process(os.getpid())
    res_heights = []
    res_sizes = []
    res_amounts = []

    all_amount = 3 * 2 ** (n - 1)
    # all_amount = 20000
    i = 0
    j = 0
    k = 0
    try:
        print(f"[!!!] Start the process for n={n}")
        for el in permute('abc', repeat=n):
            if k == 20000:
                with open(f'task2_results/task2_v2_n{n}.txt', 'w') as file:
                    file.write(f'res_heights: {res_heights}\n'
                               f'res_amounts: {res_amounts}\n'
                               f'res_sizes: {res_sizes}\n'
                               f'cur_i: {i}'
                               )
                    print(f"\n[***] SAVED TO task2_v2_n{n}.txt")
                k = 0
            if memory.value > MEMORY_SIZE // 3:
                AutomataGroupElement.clear_memory()
                _generate_H3()

            tmp_el = from_string(el)
            res_heights.append(tmp_el.tree.height())
            res_amounts.append(tmp_el.tree.vert_amount())
            res_sizes.append(tmp_el.tree.size())

            j += 1
            i += 1
            k += 1

            with done.get_lock():
                done.value = i / all_amount * 100
            with memory.get_lock():
                memory.value = _process.memory_info()[0]

    except Exception as exc:
        print("[ERROR]", exc)
        with open(f'task2_results/task2_v2_n{n}.txt', 'w') as file:
            file.write(f'res_heights: {res_heights}\n'
                       f'res_amounts: {res_amounts}\n'
                       f'res_sizes: {res_sizes}\n'
                       f'cur_i: {i}'
                       )
            print(f"[***] SAVED TO task2_v2_n{n}.txt")


Process(target=task, args=(20, done1, memory1), daemon=True).start()
Process(target=task, args=(21, done2, memory2), daemon=True).start()
Process(target=task, args=(22, done3, memory3), daemon=True).start()

obs = Process(target=info)
obs.start()
obs.join()
