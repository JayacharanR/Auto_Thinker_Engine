"""Browser-compatible, streaming video output with bounded memory."""

import av
import numpy as np


class VideoWriter:
    def __init__(self, path, width, height, fps=10):
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "23", "preset": "ultrafast", "profile": "baseline"}
        self.closed = False

    def write(self, rgb):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if not self.closed:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            self.closed = True
