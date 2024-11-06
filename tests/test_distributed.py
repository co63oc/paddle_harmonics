# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import os
import unittest

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F  # noqa
from parameterized import parameterized

import paddle_harmonics as harmonics
import paddle_harmonics.distributed as thd
from paddle_harmonics.utils import paddle_aux  # noqa


class TestDistributedSphericalHarmonicTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # set up distributed
        cls.world_rank = int(os.getenv("WORLD_RANK", 0))
        cls.grid_size_h = int(os.getenv("GRID_H", 1))
        cls.grid_size_w = int(os.getenv("GRID_W", 1))
        cls.world_size = cls.grid_size_h * cls.grid_size_w

        if paddle.device.cuda.device_count():
            if cls.world_rank == 0:
                print("Running test on GPU")
            local_rank = cls.world_rank % paddle.device.cuda.device_count()
            cls.place = paddle.CUDAPlace(int(f"cuda:{local_rank}".replace("cuda:", "")))
            paddle.seed(seed=333)
            proc_backend = "nccl"
        else:
            if cls.world_rank == 0:
                print("Running test on CPU")
            cls.place = paddle.CPUPlace()
            proc_backend = "gloo"  # noqa
        cls.device = cls.place
        paddle.seed(seed=333)

        dist.init_parallel_env()

        cls.wrank = cls.world_rank % cls.grid_size_w
        cls.hrank = cls.world_rank // cls.grid_size_w

        # now set up the comm groups:
        # set default
        cls.w_group = None
        cls.h_group = None

        # do the init
        wgroups = []
        for w in range(0, cls.world_size, cls.grid_size_w):
            start = w
            end = w + cls.grid_size_w
            wgroups.append(list(range(start, end)))

        if cls.world_rank == 0:
            print("w-groups:", wgroups)
        for grp in wgroups:
            if len(grp) == 1:
                continue
            tmp_group = dist.new_group(ranks=grp)
            if cls.world_rank in grp:
                cls.w_group = tmp_group

        # transpose:
        hgroups = [sorted(list(i)) for i in zip(*wgroups)]

        if cls.world_rank == 0:
            print("h-groups:", hgroups)
        for grp in hgroups:
            if len(grp) == 1:
                continue
            tmp_group = dist.new_group(ranks=grp)
            if cls.world_rank in grp:
                cls.h_group = tmp_group

        # set seed
        paddle.seed(seed=333)

        if cls.world_rank == 0:
            print(
                f"Running distributed tests on grid H x W = {cls.grid_size_h} x {cls.grid_size_w}"
            )

        # initializing sht
        thd.init(cls.h_group, cls.w_group)

    def _split_helper(self, tensor):
        with paddle.no_grad():
            # split in W
            tensor_list_local = thd.split_tensor_along_dim(
                tensor, dim=-1, num_chunks=self.grid_size_w
            )
            tensor_local = tensor_list_local[self.wrank]

            # split in H
            tensor_list_local = thd.split_tensor_along_dim(
                tensor_local, dim=-2, num_chunks=self.grid_size_h
            )
            tensor_local = tensor_list_local[self.hrank]

        return tensor_local

    def _gather_helper_fwd(self, tensor, B, C, transform_dist, vector):
        # we need the shapes
        l_shapes = transform_dist.l_shapes
        m_shapes = transform_dist.m_shapes

        # gather in W
        if self.grid_size_w > 1:
            if vector:
                gather_shapes = [(B, C, 2, l_shapes[self.hrank], m) for m in m_shapes]
            else:
                gather_shapes = [(B, C, l_shapes[self.hrank], m) for m in m_shapes]
            olist = [paddle.empty(shape, dtype=tensor.dtype) for shape in gather_shapes]
            olist[self.wrank] = tensor
            dist.all_gather(olist, tensor, group=self.w_group)
            tensor_gather = paddle.concat(olist, axis=-1)
        else:
            tensor_gather = tensor

        # gather in H
        if self.grid_size_h > 1:
            if vector:
                gather_shapes = [(B, C, 2, l, transform_dist.mmax) for l in l_shapes]
            else:
                gather_shapes = [(B, C, l, transform_dist.mmax) for l in l_shapes]
            olist = [paddle.empty(shape, dtype=tensor_gather.dtype) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = paddle.concat(olist, axis=-2)

        return tensor_gather

    def _gather_helper_bwd(self, tensor, B, C, transform_dist, vector):
        # we need the shapes
        lat_shapes = transform_dist.lat_shapes
        lon_shapes = transform_dist.lon_shapes

        # gather in W
        if self.grid_size_w > 1:
            if vector:
                gather_shapes = [(B, C, 2, lat_shapes[self.hrank], w) for w in lon_shapes]
            else:
                gather_shapes = [(B, C, lat_shapes[self.hrank], w) for w in lon_shapes]
            olist = [paddle.empty(shape, dtype=tensor.dtype) for shape in gather_shapes]
            olist[self.wrank] = tensor
            dist.all_gather(olist, tensor, group=self.w_group)
            tensor_gather = paddle.concat(olist, axis=-1)
        else:
            tensor_gather = tensor

        # gather in H
        if self.grid_size_h > 1:
            if vector:
                gather_shapes = [(B, C, 2, h, transform_dist.nlon) for h in lat_shapes]
            else:
                gather_shapes = [(B, C, h, transform_dist.nlon) for h in lat_shapes]
            olist = [paddle.empty(shape, dtype=tensor_gather.dtype) for shape in gather_shapes]
            olist[self.hrank] = tensor_gather
            dist.all_gather(olist, tensor_gather, group=self.h_group)
            tensor_gather = paddle.concat(olist, axis=-2)

        return tensor_gather

    @parameterized.expand(
        [
            [256, 512, 32, 8, "equiangular", False, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", False, 1e-9],
            [256, 512, 32, 8, "equiangular", False, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", False, 1e-9],
            [256, 512, 32, 8, "equiangular", False, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", False, 1e-9],
            [361, 720, 1, 10, "equiangular", False, 1e-6],
            [361, 720, 1, 10, "legendre-gauss", False, 1e-6],
            [256, 512, 32, 8, "equiangular", True, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", True, 1e-9],
            [256, 512, 32, 8, "equiangular", True, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", True, 1e-9],
            [256, 512, 32, 8, "equiangular", True, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", True, 1e-9],
            [361, 720, 1, 10, "equiangular", True, 1e-6],
            [361, 720, 1, 10, "legendre-gauss", True, 1e-6],
        ]
    )
    def test_distributed_sht(self, nlat, nlon, batch_size, num_chan, grid, vector, tol):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up handles
        if vector:
            forward_transform_local = harmonics.RealVectorSHT(nlat=H, nlon=W, grid=grid).to(
                self.device
            )
            forward_transform_dist = thd.DistributedRealVectorSHT(nlat=H, nlon=W, grid=grid).to(
                self.device
            )
        else:
            forward_transform_local = harmonics.RealSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            forward_transform_dist = thd.DistributedRealSHT(nlat=H, nlon=W, grid=grid).to(
                self.device
            )

        # create tensors
        if vector:
            inp_full = paddle.randn((B, C, 2, H, W), dtype=paddle.float32)
        else:
            inp_full = paddle.randn((B, C, H, W), dtype=paddle.float32)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.stop_gradient = not True
        out_full = forward_transform_local(inp_full)

        # create grad for backward
        with paddle.no_grad():
            # create full grad
            ograd_full = paddle.randn(shape=out_full.shape, dtype=out_full.dtype)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.stop_gradient = not True
        out_local = forward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = forward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with paddle.no_grad():
            out_gather_full = self._gather_helper_fwd(
                out_local, B, C, forward_transform_dist, vector
            )
            err = paddle.mean(
                paddle_aux.norm_complex(out_full - out_gather_full, p="fro", axis=(-1, -2))
                / paddle_aux.norm_complex(out_full, p="fro", axis=(-1, -2))
            )
            if self.world_rank == 0:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with paddle.no_grad():
            igrad_gather_full = self._gather_helper_bwd(
                igrad_local, B, C, forward_transform_dist, vector
            )
            err = paddle.mean(
                paddle.linalg.norm(igrad_full - igrad_gather_full, p="fro", axis=(-1, -2))
                / paddle.linalg.norm(igrad_full, p="fro", axis=(-1, -2))
            )
            if self.world_rank == 0:
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

    @parameterized.expand(
        [
            [256, 512, 32, 8, "equiangular", False, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", False, 1e-9],
            [256, 512, 32, 8, "equiangular", False, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", False, 1e-9],
            [256, 512, 32, 8, "equiangular", False, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", False, 1e-9],
            [361, 720, 1, 10, "equiangular", False, 1e-6],
            [361, 720, 1, 10, "legendre-gauss", False, 1e-6],
            [256, 512, 32, 8, "equiangular", True, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", True, 1e-9],
            [256, 512, 32, 8, "equiangular", True, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", True, 1e-9],
            [256, 512, 32, 8, "equiangular", True, 1e-9],
            [256, 512, 32, 8, "legendre-gauss", True, 1e-9],
            [361, 720, 1, 10, "equiangular", True, 1e-6],
            [361, 720, 1, 10, "legendre-gauss", True, 1e-6],
        ]
    )
    def test_distributed_isht(self, nlat, nlon, batch_size, num_chan, grid, vector, tol):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        if vector:
            forward_transform_local = harmonics.RealVectorSHT(nlat=H, nlon=W, grid=grid).to(
                self.device
            )
            backward_transform_local = harmonics.InverseRealVectorSHT(nlat=H, nlon=W, grid=grid).to(
                self.device
            )
            backward_transform_dist = thd.DistributedInverseRealVectorSHT(
                nlat=H, nlon=W, grid=grid
            ).to(self.device)
        else:
            forward_transform_local = harmonics.RealSHT(nlat=H, nlon=W, grid=grid).to(self.device)
            backward_transform_local = harmonics.InverseRealSHT(nlat=H, nlon=W, grid=grid).to(
                self.device
            )
            backward_transform_dist = thd.DistributedInverseRealSHT(nlat=H, nlon=W, grid=grid).to(
                self.device
            )

        # create tensors
        if vector:
            dummy_full = paddle.randn((B, C, 2, H, W), dtype=paddle.float32)
        else:
            dummy_full = paddle.randn((B, C, H, W), dtype=paddle.float32)
        inp_full = forward_transform_local(dummy_full)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.stop_gradient = not True
        out_full = backward_transform_local(inp_full)

        # create grad for backward
        with paddle.no_grad():
            # create full grad
            ograd_full = paddle.randn(shape=out_full.shape, dtype=out_full.dtype)

        # BWD pass
        out_full.backward(ograd_full)

        # repeat once due to known irfft bug
        inp_full.clear_grad()
        out_full = backward_transform_local(inp_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()  # noqa

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.stop_gradient = not True
        out_local = backward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = backward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()  # noqa

        #############################################################
        # evaluate FWD pass
        #############################################################
        with paddle.no_grad():
            out_gather_full = self._gather_helper_bwd(
                out_local, B, C, backward_transform_dist, vector
            )
            err = paddle.mean(
                paddle.linalg.norm(out_full - out_gather_full, p="fro", axis=(-1, -2))
                / paddle.linalg.norm(out_full, p="fro", axis=(-1, -2))
            )
            if self.world_rank == 0:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with paddle.no_grad():
            igrad_gather_full = self._gather_helper_fwd(
                igrad_local, B, C, backward_transform_dist, vector
            )
            err = paddle.mean(
                paddle_aux.norm_complex(igrad_full - igrad_gather_full, p="fro", axis=(-1, -2))
                / paddle_aux.norm_complex(igrad_full, p="fro", axis=(-1, -2))
            )
            if self.world_rank == 0:
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)


if __name__ == "__main__":
    unittest.main()
