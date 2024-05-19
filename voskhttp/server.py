import json
import logging
import os
import shlex
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from timeit import default_timer as timer
from typing import Final

from .vosk_cffi import ffi as _ffi

JA_MODEL_FOLDER: Final = "vosk-model-small-ja-0.22"
CHUNK_SIZE: Final = 4000
SAMPLE_RATE: Final = 16000.0


def open_dll():
    dlldir = os.path.abspath(os.path.dirname(__file__))
    if sys.platform == "win32":
        # We want to load dependencies too
        os.environ["PATH"] = dlldir + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dlldir)
        return _ffi.dlopen(os.path.join(dlldir, "libvosk.dll"))
    elif sys.platform == "linux":
        return _ffi.dlopen(os.path.join(dlldir, "libvosk.so"))
    else:
        raise TypeError("Unsupported platform")


_c = open_dll()


class Model:
    def __init__(self, model_path: Path):
        if model_path.exists():
            self._handle = _c.vosk_model_new(bytes(model_path))
        else:
            raise FileNotFoundError(model_path)

    def __del__(self):
        _c.vosk_model_free(self._handle)


class KaldiRecognizer:
    def __init__(self, model: Model, sample_rate: float):
        self._handle = _c.vosk_recognizer_new(model._handle, sample_rate)
        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a recognizer")

    def __del__(self):
        _c.vosk_recognizer_free(self._handle)

    def SetWords(self, enable_words):
        _c.vosk_recognizer_set_words(self._handle, 1 if enable_words else 0)

    def AcceptWaveform(self, data):
        res = _c.vosk_recognizer_accept_waveform(self._handle, data, len(data))
        if res < 0:
            raise Exception("Failed to process waveform")
        return res

    def Result(self):
        return _ffi.string(_c.vosk_recognizer_result(self._handle)).decode("utf-8")

    def PartialResult(self):
        return _ffi.string(_c.vosk_recognizer_partial_result(self._handle)).decode("utf-8")

    def FinalResult(self):
        return _ffi.string(_c.vosk_recognizer_final_result(self._handle)).decode("utf-8")

    def Reset(self):
        return _c.vosk_recognizer_reset(self._handle)


def recognize_stream(rec: KaldiRecognizer, stream):
    tot_samples = 0
    result = []

    while True:
        data = stream.stdout.read(CHUNK_SIZE)

        if len(data) == 0:
            break

        tot_samples += len(data)
        if rec.AcceptWaveform(data):
            jres = json.loads(rec.Result())
            logging.info(jres)
            result.append(jres)
        else:
            jres = json.loads(rec.PartialResult())
            if jres["partial"] != "":
                logging.info(jres)

    jres = json.loads(rec.FinalResult())
    result.append(jres)

    return result, tot_samples


def resample_ffmpeg(infile):
    cmd = shlex.split("ffmpeg -nostdin -loglevel quiet "
                      "-i \'{}\' -ar {} -ac 1 -f s16le -".format(str(infile), SAMPLE_RATE))
    stream = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return stream


def format_result(result):
    monologues = {"schemaVersion": "2.0", "monologues": [], "text": []}
    for part in result:
        if part["text"] != "":
            monologues["text"] += [part["text"]]
    for _, res in enumerate(result):
        if "result" not in res:
            continue
        monologue = {
            "speaker": {"id": "unknown", "name": None},
            "start": res["result"][0]["start"],
            "end": res["result"][-1]["end"],
            "terms": [
                {"confidence": t["conf"], "start": t["start"], "end": t["end"], "text": t["word"], "type": "WORD"}
                for t in res["result"]
            ]
        }
        monologues["monologues"].append(monologue)
    return json.dumps(monologues)


class VoskRequestHandler(BaseHTTPRequestHandler):
    model: Model | None = None

    def __init__(self, model: Model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def recognize(self, input_path):
        assert self.model, "No valid model given"

        logging.info("Recognizing {}".format(input_path))
        start_time = timer()

        try:
            stream = resample_ffmpeg(input_path)
        except FileNotFoundError as e:
            logging.error(e, "Missing FFMPEG, please install and try again")
            return
        except Exception as e:
            logging.info(e)
            return

        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)
        result, tot_samples = recognize_stream(rec, stream)
        if tot_samples == 0:
            return

        processed_result = format_result(result)

        elapsed = timer() - start_time
        logging.warning("Execution time: {:.3f} sec; "
                        "xRT {:.3f}".format(elapsed, float(elapsed) * (2 * SAMPLE_RATE) / tot_samples))

        return processed_result

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        path = Path(self.rfile.read(length).decode())
        if path.exists():
            logging.info(f"Request POST ({path.as_posix()}) - Response 200 OK")
            payload = self.recognize(str(path))
            if payload is None:
                logging.warning(f"Invalid file: {path.as_posix()}")
                payload = json.dumps({"message": f"Invalid file: {path.as_posix()}"})
                self.send_response(400)
            else:
                self.send_response(200)
        else:
            logging.warning(f"Request POST ({path.as_posix()}) - Response 400 Bad Request")
            payload = json.dumps({"message": f"File not found: {path.as_posix()}"})
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        payload = payload.encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class VoskServer(HTTPServer):
    hostname = "localhost"
    port = 8004

    model = None

    def __init__(self, hostname: str = None, port: int = None):
        if hostname is not None:
            self.hostname = hostname
        if port is not None:
            self.port = port
        ja_model_path = Path(__file__).parent / "models" / JA_MODEL_FOLDER
        self.model = Model(ja_model_path)

        def handler(*args, **kwargs):
            return VoskRequestHandler(self.model, *args, **kwargs)

        super().__init__((self.hostname, self.port), handler)

    def serve_forever(self, *args, **kwargs):
        print(f"Starting server on {self.hostname}:{self.port}")
        super().serve_forever(*args, **kwargs)


def run(hostname: str = None, port: int = None):
    server = VoskServer(hostname, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
