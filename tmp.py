#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 22.11.2019
# by David Zashkolny
# email: davendiy@gmail.com
#
#
# def check_el(str_repr):
#     global counter
#     el_group = GroupElement(str_repr)
#     if el_group.is_primitive():
#         logging.info(el_group)
#
#     with counter.get_lock():
#         counter.value += 1
#
#
# def observer():
#     global counter
#     while True:
#         logging.info(f"done: {counter.value}")
#         time.sleep(1)
#
# n = 20
#
# with ProcessPoolExecutor() as executor:
#     obs = Process(target=observer)
#     obs.start()
#     for el_list in permute('abc', repeat=n):
#         el_str = ''.join(el_list)
#         executor.submit(check_el, el_str)
#     obs.join()
