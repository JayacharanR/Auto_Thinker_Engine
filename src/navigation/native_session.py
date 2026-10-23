"""Release the owned native client before terminating its simulator process.

Call a scoped episode function that returns plain data only. World/actor handles
must not escape that function. This does not change sensor, physics or controls.
"""

import gc
from contextlib import contextmanager


class ClientLease:
    """Only this proxy owns the Python Client; callers can release it explicitly."""

    def __init__(self, native_client):
        self._native_client = native_client

    def __getattr__(self, name):
        client = self._native_client
        if client is None:
            raise RuntimeError("Owned native client already released")
        return getattr(client, name)

    def release(self):
        self._native_client = None
        gc.collect()


@contextmanager
def owned_native_session(server):
    """Keep the engine alive while the last scoped Client is released.

    Native callback failures cannot be caught reliably in Python; use an outer
    process supervisor as well. A unit test is not native lifecycle validation.
    """
    lease = ClientLease(server.__enter__())
    try:
        yield server, lease
    finally:
        try:
            lease.release()
        finally:
            server.close()
