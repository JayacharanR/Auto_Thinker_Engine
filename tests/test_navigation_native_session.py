"""Client/callback ownership ends while its server is still alive."""

import weakref

import pytest

from src.navigation.native_session import owned_native_session


def test_last_client_reference_is_released_before_server_close_even_after_error():
    events = []

    class NativeClient:
        def query(self):
            return "live"

        def __del__(self):
            events.append("client_released")

    class Server:
        def __enter__(self):
            client = NativeClient()
            self.reference = weakref.ref(client)
            return client

        def close(self):
            assert self.reference() is None
            events.append("server_closed")

    server = Server()
    with pytest.raises(ValueError, match="episode failed"):
        with owned_native_session(server) as (owner, client):
            assert owner is server and client.query() == "live"
            raise ValueError("episode failed")
    assert events == ["client_released", "server_closed"]
    with pytest.raises(RuntimeError, match="already released"):
        client.query()
