# -*- coding: utf-8 -*-

import itertools
import os
import sys
import threading
import time

from collections import namedtuple
from .handlers import notebook_dir   # noqa
from tensorboard import default, program
from tensorboard.backend import application   # noqa

sys.argv = ["tensorboard"]
TensorBoardInstance = namedtuple(
    'TensorBoardInstance', ['name', 'logdir', 'tb_app', 'thread'])


def create_tb_app(logdir, reload_interval, purge_orphaned_data):
    argv = [
        "--logdir", logdir,
        "--reload_interval", str(reload_interval),
        "--purge_orphaned_data", str(purge_orphaned_data),
    ]
    tensorboard = program.TensorBoard(
        default.PLUGIN_LOADERS,
        default.get_assets_zip_provider())
    tensorboard.configure(argv)
    return application.standard_tensorboard_wsgi(
        tensorboard.flags,
        tensorboard.plugin_loaders,
        tensorboard.assets_zip_provider)


def start_reloading_multiplexer(multiplexer, path_to_run, reload_interval):
    def _ReloadForever():
        current_thread = threading.currentThread()
        while not current_thread.stop:
            application.reload_multiplexer(multiplexer, path_to_run)
            current_thread.reload_time = time.time()
            time.sleep(reload_interval)
    thread = threading.Thread(target=_ReloadForever)
    thread.reload_time = None
    thread.stop = False
    thread.daemon = True
    thread.start()
    return thread


# Wrapper around TensorBoardWSGIApp from TensorBoard's application.py. Includes
# a call to the TensorBoard instance manager.
def TensorBoardWSGIApp(logdir, plugins, multiplexer,
                       reload_interval, using_db, path_prefix=""):
    path_to_run = application.parse_event_files_spec(logdir)
    if reload_interval >= 0:
        thread = start_reloading_multiplexer(
            multiplexer, path_to_run, reload_interval)
    else:
        application.reload_multiplexer(multiplexer, path_to_run)
        thread = None
    tb_app = application.TensorBoardWSGI(plugins, using_db, path_prefix)
    manager.add_instance(logdir, tb_app, thread)
    return tb_app


application.TensorBoardWSGIApp = TensorBoardWSGIApp


class TensorBoardManger(dict):

    def __init__(self):
        self._logdir_dict = {}

    def _next_available_name(self):
        for n in itertools.count(start=1):
            name = "%d" % n
            if name not in self:
                return name

    def new_instance(self, logdir, reload_interval):
        if not os.path.isabs(logdir) and notebook_dir:
            logdir = os.path.join(notebook_dir, logdir)

        if logdir not in self._logdir_dict:
            purge_orphaned_data = True
            reload_interval = reload_interval or 30
            create_tb_app(
                logdir=logdir, reload_interval=reload_interval,
                purge_orphaned_data=purge_orphaned_data)

        return self._logdir_dict[logdir]

    def add_instance(self, logdir, tb_application, thread):
        name = self._next_available_name()
        instance = TensorBoardInstance(name, logdir, tb_application, thread)
        self[name] = instance
        self._logdir_dict[logdir] = instance

    def terminate(self, name, force=True):
        if name in self:
            instance = self[name]
            if instance.thread is not None:
                instance.thread.stop = True
            del self[name], self._logdir_dict[instance.logdir]
        else:
            raise Exception("There's no TensorBoard instance named %s" % name)


manager = TensorBoardManger()
