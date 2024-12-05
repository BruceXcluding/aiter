# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: core.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-29 15:58:57
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-04 21:55:44
# @Description: This is description.

import os
import sys
import shutil
import time
import importlib
from typing import List, Optional
from torch.utils import cpp_extension
import logging
logger = logging.getLogger("ater")

PYTHON = sys.executable
this_dir = os.path.dirname(os.path.abspath(__file__))
ATER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
ATER_CSRC_DIR = f'{ATER_ROOT_DIR}/csrc'
CK_DIR = os.environ.get(
    "CK_DIR", f"{ATER_ROOT_DIR}/3rdparty/composable_kernel")
bd_dir = f"{this_dir}/build"
# copy ck to build, thus hippify under bd_dir
shutil.copytree(CK_DIR, f'{bd_dir}/ck', dirs_exist_ok=True)
CK_DIR = f'{bd_dir}/ck'


def validate_and_update_archs():
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a",
                     "gfx940", "gfx941", "gfx942", "gfx1100"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"


def check_and_set_ninja_worker():
    max_num_jobs_cores = int(max(1, os.cpu_count()*0.8))
    if int(os.environ.get("MAX_JOBS", '1')) < max_num_jobs_cores:
        import psutil
        # calculate the maximum allowed NUM_JOBS based on free memory
        free_memory_gb = psutil.virtual_memory().available / \
            (1024 ** 3)  # free memory in GB
        # each JOB peak memory cost is ~8-9GB when threads = 4
        max_num_jobs_memory = int(free_memory_gb / 9)

        # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
        max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
        max_jobs = str(max_jobs)
        os.environ["MAX_JOBS"] = max_jobs


def rename_cpp_to_cu(els, dst):
    def do_rename_and_mv(name, src, dst, ret):
        newName = name
        if name.endswith(".cpp") or name.endswith(".cu"):
            newName = name.replace(".cpp", ".cu")
            ret.append(f'{dst}/{newName}')
        shutil.copy(f'{src}/{name}', f'{dst}/{newName}')
    ret = []
    for el in els:
        if not os.path.exists(el):
            continue
        if os.path.isdir(el):
            for entry in os.listdir(el):
                if os.path.isdir(f'{el}/{entry}'):
                    continue
                do_rename_and_mv(entry, el, dst, ret)
        else:
            do_rename_and_mv(os.path.basename(el),
                             os.path.dirname(el), dst, ret)
    return ret


def compile_ops(
    srcs: List[str],
    md_name: str,
    fc_name: Optional[str] = None,
    flags_extra_cc: List[str] = [],
    flags_extra_hip: List[str] = [],
    extra_ldflags=None,
    extra_include: List[str] = [],
    verbose=False,
    blob_gen_cmd=''
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            loadName = fc_name
            if fc_name is None:
                loadName = func.__name__

            try:
                if importlib.util.find_spec('ater_') is not None:
                    import ater_
                    if hasattr(ater_, loadName):
                        return getattr(ater_, loadName)(*args, **kwargs)
                module = importlib.import_module(f'{__package__}.{md_name}')
            except Exception as e:
                op_dir = f'{bd_dir}/{md_name}'
                logger.info(f'start build [{md_name}] under {op_dir}')

                startTS = time.perf_counter()
                opbd_dir = f'{op_dir}/build'
                src_dir = f'{op_dir}/build/srcs'
                os.makedirs(src_dir, exist_ok=True)
                sources = rename_cpp_to_cu(srcs, src_dir)

                flags_cc = ["-O3", "-std=c++17",
                            "-mllvm", "-enable-post-misched=0",
                            "-mllvm", "-amdgpu-early-inline-all=true",
                            "-mllvm", "-amdgpu-function-calls=false",
                            "-mllvm", "--amdgpu-kernarg-preload-count=16",
                            "-mllvm", "-amdgpu-coerce-illegal-types=1",
                            # "-v", "--save-temps",
                            "-Wno-unused-result",
                            "-Wno-switch-bool",
                            "-Wno-vla-cxx-extension",
                            "-Wno-undefined-func-template",
                            "-Wno-switch-bool",
                            ]
                flags_hip = [
                    "-DUSE_PROF_API=1",
                    "-D__HIP_PLATFORM_HCC__=1",
                    "-D__HIP_PLATFORM_AMD__=1",
                    "-U__HIP_NO_HALF_CONVERSIONS__",
                    "-U__HIP_NO_HALF_OPERATORS__",
                ] + flags_cc
                flags_cc += flags_extra_cc
                flags_hip += flags_extra_hip
                validate_and_update_archs()
                check_and_set_ninja_worker()

                if blob_gen_cmd:
                    blob_dir = f"{op_dir}/blob"
                    os.makedirs(blob_dir, exist_ok=True)
                    os.system(f'{PYTHON} {blob_gen_cmd.format(blob_dir)}')

                    sources += rename_cpp_to_cu([blob_dir], src_dir)
                extra_include_paths = [
                    f"{CK_DIR}/include",
                    f"{CK_DIR}/library/include",
                    f"{ATER_CSRC_DIR}/include",
                ]+extra_include

                module = cpp_extension.load(
                    md_name,
                    sources,
                    extra_cflags=flags_cc,
                    extra_cuda_cflags=flags_hip,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=opbd_dir,
                    verbose=verbose or int(os.getenv("ATER_LOG_MORE")) > 0,
                    with_cuda=True,
                    is_python_module=True,
                )
                shutil.copy(f'{opbd_dir}/{md_name}.so', f'{this_dir}')
                logger.info(
                    f'finish build [{md_name}], cost {time.perf_counter()-startTS:.8f}s')

            return getattr(module, loadName)(*args, **kwargs)
        return wrapper
    return decorator
