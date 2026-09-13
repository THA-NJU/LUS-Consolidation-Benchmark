"""Track native checkpoint bytes through buffers, copies, and atomic renames."""
import hashlib
import os
from pathlib import Path


class CheckpointTracker:
    def __init__(self, report):
        self.report = report
        self.saved_digests = set()

    @staticmethod
    def name(target):
        value = target if isinstance(target, (str, os.PathLike)) else getattr(target, 'name', None)
        return str(Path(value).absolute()) if isinstance(value, (str, os.PathLike)) else None

    @classmethod
    def digest(cls, target):
        if hasattr(target, 'getbuffer'):
            return hashlib.sha256(target.getbuffer()).hexdigest()
        path = cls.name(target)
        if not path or not Path(path).is_file():
            return None
        # File-backed torch.save targets can still be buffered when it returns.
        if hasattr(target, 'flush') and not getattr(target, 'closed', False):
            target.flush()
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def saved(self, target):
        digest = self.digest(target)
        if digest is not None:
            self.saved_digests.add(digest)
            self.report['saved_checkpoints'].append(self.name(target) or 'buffer:sha256:' + digest)

    def loaded(self, target, digest):
        if digest is not None and digest in self.saved_digests:
            self.report['reloaded_checkpoints'].append(self.name(target) or 'buffer:sha256:' + digest)
