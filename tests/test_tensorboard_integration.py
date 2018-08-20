# -*- coding:utf-8 -*-

import sys
import time
import logging
import json

import pytest
from tornado.testing import AsyncHTTPTestCase


@pytest.fixture(scope="session")
def nb_app():
    sys.argv = ["--port=6005", "--ip=127.0.0.1", "--no-browser", "--debug"]
    from notebook.notebookapp import NotebookApp
    app = NotebookApp()
    app.log_level = logging.DEBUG
    app.ip = '127.0.0.1'
    # TODO: Add auth check tests
    app.token = ''
    app.password = ''
    app.disable_check_xsrf = True
    app.initialize()
    return app.web_app


class TestJupyterExtension(AsyncHTTPTestCase):

    @pytest.fixture(autouse=True)
    def init_jupyter(self, tf_logs, nb_app, tmpdir_factory):
        self.app = nb_app
        self.log_dir = tf_logs
        self.tmpdir_factory = tmpdir_factory

    def get_app(self):
        return self.app

    def test_tensorboard(self):

        content = {"logdir": self.log_dir}
        content_type = {"Content-Type": "application/json"}
        response = self.fetch(
            '/api/tensorboard',
            method='POST',
            body=json.dumps(content),
            headers=content_type)

        response = self.fetch('/api/tensorboard')
        instances = json.loads(response.body.decode())
        assert len(instances) > 0

        response = self.fetch('/api/tensorboard/1')
        instance = json.loads(response.body.decode())
        instance2 = None
        for inst in instances:
            if inst["name"] == instance["name"]:
                instance2 = inst
        assert instance == instance2

        response = self.fetch('/tensorboard/1/#graphs')
        assert response.code == 200

        response = self.fetch('/tensorboard/1/data/plugins_listing')
        plugins_list = json.loads(response.body.decode())
        assert plugins_list["graphs"]
        assert plugins_list["scalars"]

        response = self.fetch(
            '/api/tensorboard/1',
            method='DELETE')
        assert response.code == 204

        response = self.fetch('/api/tensorboard/1')
        error_msg = json.loads(response.body.decode())
        assert error_msg["message"].startswith(
            "TensorBoard instance not found:")

    def test_instance_reload(self):
        content = {"logdir": self.log_dir, "reload_interval": 4}
        content_type = {"Content-Type": "application/json"}
        response = self.fetch(
            '/api/tensorboard',
            method='POST',
            body=json.dumps(content),
            headers=content_type)
        instance = json.loads(response.body.decode())
        assert instance is not None
        name = instance["name"]
        reload_time = instance["reload_time"]

        time.sleep(5)
        response = self.fetch('/api/tensorboard/{}'.format(name))
        instance2 = json.loads(response.body.decode())
        assert instance2["reload_time"] != reload_time
