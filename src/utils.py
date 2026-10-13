import os
import random

import numpy as np

from config import PROJECT_ROOT


def keep_full_speed():

    # Windows 11 treats a hidden background process as low priority
    # ("power throttling") and moves it to the slow efficiency cores,
    # which made training epochs ~4x slower. Opt this process out and
    # raise its priority slightly. No admin rights needed; no effect on
    # results, only on speed. Does nothing on other systems.
    if os.name != "nt":
        return

    try:
        import ctypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong),
                        ("ControlMask", ctypes.c_ulong),
                        ("StateMask", ctypes.c_ulong)]

        kernel32 = ctypes.windll.kernel32

        # 64-bit handles: declare pointer types, or the calls fail silently
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

        process = kernel32.GetCurrentProcess()

        # Execution-speed and timer-resolution throttling off
        state = PowerThrottlingState(1, 0x1 | 0x4, 0)
        kernel32.SetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state))

        ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
        kernel32.SetPriorityClass(process, ABOVE_NORMAL_PRIORITY_CLASS)

    except (AttributeError, OSError):
        pass


# Every project script imports utils, so each new process opts out
keep_full_speed()


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    # PyTorch is optional (dataset inspection runs without it)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)


def relative(path):

    # Path relative to the project root, with "/" so saved file lists
    # are identical on every operating system
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
