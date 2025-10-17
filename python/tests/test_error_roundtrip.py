# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Regression tests for Python exception round-tripping.

These tests currently fail because Python exceptions raised from custom data
sources are wrapped as ``PyDataFusionError`` when converted back from the Rust
bindings. The desired behaviour is for the original Python exception type and
traceback to be preserved so that developers can inspect the failure directly.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from datafusion.catalog import Table


class ExplodingInMemoryDataset(ds.InMemoryDataset):
    """In-memory dataset whose fragment iterator raises ``ValueError``."""

    def __init__(self, message: str) -> None:
        batch = pa.record_batch([pa.array([1], type=pa.int32())], names=["value"])
        super().__init__([batch])
        self._message = message

    def get_fragments(self, filter=None):  # type: ignore[override]
        raise ValueError(self._message)


def test_dataset_fragment_exception_round_trips() -> None:
    """Python errors from dataset fragments should propagate untouched."""

    ctx = datafusion.SessionContext()
    dataset = ExplodingInMemoryDataset("dataset fragment boom")
    ctx.register_dataset("fail", dataset)

    with pytest.raises(ValueError, match="dataset fragment boom"):
        ctx.sql("SELECT * FROM fail").collect()


def test_table_provider_exception_round_trips() -> None:
    """Python errors from table providers should propagate untouched."""

    ctx = datafusion.SessionContext()
    dataset = ExplodingInMemoryDataset("table provider boom")
    ctx.register_table("fail", Table.from_dataset(dataset))

    with pytest.raises(ValueError, match="table provider boom"):
        ctx.sql("SELECT * FROM fail").collect()
