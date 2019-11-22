#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 22.11.2019
# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

from source import GroupElement, AutomataGroupElement
from multiprocessing import Value, Queue, Process
from logger import logger
import time


# interesting = Array('i', [])
done = Value('i', 0)
filled = Value('i', 0)


j = 0

CHUNK = 1024 * 1000
FIRST = AutomataGroupElement.DEFINED_ELEMENTS.copy()

queue = Queue()


def permute(seq, repeat):
    if repeat == 1:
        for el in seq:
            yield [el]
    elif repeat < 1:
        yield []
    else:
        for prev in permute(seq, repeat-1):
            for el in seq:
                if prev[-1] == el:
                    continue
                yield prev + [el]


def check_el(str_repr):
    el_group = GroupElement(str_repr)
    return el_group.is_trivial()


def worker():
    i = 0
    while True:
        i += 1
        target = queue.get()
        try:
            if len(AutomataGroupElement.DEFINED_ELEMENTS) > CHUNK:
                del AutomataGroupElement.DEFINED_ELEMENTS
                AutomataGroupElement.DEFINED_ELEMENTS = FIRST.copy()

            if check_el(target):
                logger.critical(target)
            with done.get_lock():
                done.value += 1
            if i == 100:
                i = 0
                with filled.get_lock():
                    filled.value = len(AutomataGroupElement.DEFINED_ELEMENTS)
        except Exception as e:
            logger.error(e)
            break


def observer():
    global done
    while True:
        logger.info(f"[*] done: {done.value};    "
                    f"memory usage: absolute={filled.value}, relative={filled.value * 100 / CHUNK}%;    "
                    f"queue len: {queue.qsize()}")
        time.sleep(10)



n = 20

Process(target=worker, daemon=True).start()
Process(target=worker, daemon=True).start()
Process(target=worker, daemon=True).start()

obs = Process(target=observer)
obs.start()

PREV_RES = 10510641

for el_list in permute('abc', repeat=n):
    el_str = ''.join(el_list)
    if done.value < PREV_RES:
        done.value += 1
    else:
        queue.put(el_str)

while not queue.empty():
    time.sleep(10)

logger.info('[!!!] n = 21 started.')
for el_list in permute('abc', repeat=n+1):
    el_str = ''.join(el_list)
    if done.value < PREV_RES:
        done.value += 1
    else:
        queue.put(el_str)
        if queue.qsize() > CHUNK:
            time.sleep(10)

while not queue.empty():
    time.sleep(10)

logger.info('[!!!] n = 22 started.')

for el_list in permute('abc', repeat=n+2):
    el_str = ''.join(el_list)
    if done.value < PREV_RES:
        done.value += 1
    else:
        queue.put(el_str)
        if queue.qsize() > CHUNK:
            time.sleep(10)
obs.join()
