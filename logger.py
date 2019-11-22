#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 03.11.2019
# by David Zashkolny
# email: davendiy@gmail.com

import logging

# Create a custom logger
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# Create handlers
i_handler = logging.FileHandler('loop.log')
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('critical.log')

# all the information will be written to the loop.log
# all the warning and error will be written to the stdout
# all the error will be written to the error.log
i_handler.setLevel(logging.INFO)
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.CRITICAL)

# Create formatters and add it to handlers
i_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(message)s')

i_handler.setFormatter(i_format)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(i_handler)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# logger.info('This is an info')         # test
# logger.warning('This is a warning')
# logger.error('This is an error')
