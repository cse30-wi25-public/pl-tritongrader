import logging

from typing import Callable, List, Tuple
import importlib.util
import uuid
import os

from tritongrader.test_case.test_case_base import TestCaseBase, TestResultBase
from tempfile import TemporaryDirectory
from tritongrader.runner import CommandRunner
import shlex
import subprocess

logger = logging.getLogger("tritongrader.test_case")


class RealtimeTestResult(TestResultBase):
    def __init__(self):
        super().__init__()
        self.exit_status: int | None = None
        self.stderr: str = ""
        self.stdout: str = ""


class RealtimeTestCase(TestCaseBase):
    def __init__(
        self,
        generator: str,
        sandbox: TemporaryDirectory,
        sandbox_reference: TemporaryDirectory,
        id: int = -1,
        name: str = "Test Case",
        point_value: float = 1,
        timeout: float = TestCaseBase.DEFAULT_TIMEOUT,
        interpreter: str | None = None,
        hidden: bool = False,
        hidden_msg: str = "hidden tests"
    ):
        super().__init__(name, point_value, timeout, hidden)
        self.generator: str = generator
        self.sandbox: TemporaryDirectory = sandbox
        self.sandbox_reference: TemporaryDirectory = sandbox_reference
        self.hidden_msg: str = hidden_msg
        self.name: str = name
        self.interpreter: str | None = interpreter
        self.data: dict = {"id": id, "name": name, "point_value": point_value, "timeout": timeout}
        self.added_files: List[str] = []
        self.made_dirs: List[str] = []

        self.result: RealtimeTestResult = RealtimeTestResult()
        self.runner: CommandRunner | None = None

    @property
    def expected_stdout(self):
        if not self.exp_stdout_path:
            return None
        with open(self.exp_stdout_path, "r") as fp:
            return fp.read()

    @property
    def expected_stderr(self):
        if not self.exp_stderr_path:
            return None
        with open(self.exp_stderr_path, "r") as fp:
            return fp.read()

    @property
    def actual_stdout(self) -> str:
        if not self.runner:
            raise Exception("no runner initialized")
        return self.runner.stdout

    @property
    def actual_stderr(self) -> str:
        if not self.runner:
            raise Exception("no runner initialized")
        return self.runner.stderr

    @property
    def test_input(self):
        return self.data["stdin"]

    def load_generate_func(self):
        spec = importlib.util.spec_from_file_location(f"module.{self.name}", self.generator)
        if spec is None:
            logger.error(f"Failed to load generator {self.generator}")
            raise Exception(f"Failed to load generator {self.generator}")

        if spec.loader is None:
            logger.error(f"Failed to load generator {self.generator}")
            raise Exception(f"Failed to load generator {self.generator}")

        module = importlib.util.module_from_spec(spec)
        if module is None:
            logger.error(f"Failed to load generator {self.generator}")
            raise Exception(f"Failed to load generator {self.generator}")

        spec.loader.exec_module(module)

        if not hasattr(module, "generate"):
            logger.error(f"Failed to load generator() in {self.generator}")
            raise Exception(f"Failed to load generator() in {self.generator}")

        generate_func = getattr(module, "generate")

        return generate_func

    def copy2sandbox(self, sandbox):
        with open(os.path.join(sandbox.name, self.filename_stdin), "w") as f:
            self.added_files.append(os.path.join(sandbox.name, self.filename_stdin))
            f.write(self.data["stdin"])

        if "file" in self.data:
            for filename, content in self.data["file"]:
                fullpath = os.path.join(sandbox.name, filename)
                dir_part = os.path.dirname(fullpath)
                if dir_part:
                    self.made_dirs.append(dir_part)
                    os.makedirs(dir_part, exist_ok=True)
                with open(fullpath, "w") as f:
                    self.added_files.append(fullpath)
                    f.write(content)

    def get_execute_command(self, sandbox, capture_output=False):
        filename = os.path.join(sandbox.name, self.data["argv"][0])
        args = self.data["argv"][1:]
        stdin = os.path.join(sandbox.name, self.filename_stdin)
        shell_command = f"{shlex.join([filename] + args)} < {stdin}"
        if capture_output:
            stdout = os.path.join(sandbox.name, self.filename_stdout)
            stderr = os.path.join(sandbox.name, self.filename_stderr)
            self.added_files.append(stdout)
            self.added_files.append(stderr)
            shell_command += f" > {stdout} 2> {stderr}"
        return shell_command

    def write_out_err(self):
        self.exp_stdout_path = os.path.join(self.sandbox_reference.name, self.filename_stdout)
        self.exp_stderr_path = os.path.join(self.sandbox_reference.name, self.filename_stderr)
        self.added_files.append(self.exp_stdout_path)
        self.added_files.append(self.exp_stderr_path)
        if "stdout" in self.data and "stderr" in self.data and "exitcode" in self.data:
            with open(self.exp_stdout_path, "w") as f:
                f.write(self.data["stdout"])
            with open(self.exp_stderr_path, "w") as f:
                f.write(self.data["stderr"])
            self.exp_exit_status = self.data["exitcode"]
        else:
            cmd = self.get_execute_command(self.sandbox_reference, capture_output=True)
            try:
                cwd = os.getcwd()
                os.chdir(self.sandbox_reference.name)
                runner = CommandRunner(
                    command=cmd,
                    capture_output=False,
                    text=True,
                    timeout=self.timeout,
                    print_command=False,
                    interpreter=self.interpreter,
                )
                runner.run()
                os.chdir(cwd)
                if "exitcode" not in self.data:
                    self.exp_exit_status = runner.exit_status
            except subprocess.TimeoutExpired:
                logger.info(f"Reference {self.name} timed out (limit={self.timeout}s)!")
                raise Exception(f"Reference {self.name} timed out (limit={self.timeout}s)!")

    def __del__(self):
        for fpath in self.added_files:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError as e:
                    logger.warning(f"Failed to remove {fpath}: {e}")
        for dpath in reversed(self.made_dirs):
            if os.path.isdir(dpath):
                try:
                    os.rmdir(dpath)
                except OSError as e:
                    logger.warning(f"Failed to remove {dpath}: {e}")


    def execute(self):
        self.result.has_run = True

        gen = self.load_generate_func()

        gen(self.data)

        if not "stdin" in self.data:
            logger.error(f"Failed to load stdin in {self.generator}")
            raise Exception(f"Failed to load stdin in {self.generator}")
        if not "argv" in self.data:
            logger.error(f"Failed to load argv in {self.generator}")
            raise Exception(f"Failed to load argv in {self.generator}")

        self.command = shlex.join(self.data["argv"])

        self.filename_stdin = str(uuid.uuid4()).replace("-", "")
        self.filename_stdout = str(uuid.uuid4()).replace("-", "")
        self.filename_stderr = str(uuid.uuid4()).replace("-", "")
        self.copy2sandbox(self.sandbox)
        self.copy2sandbox(self.sandbox_reference)
        self.write_out_err()

        try:
            cwd = os.getcwd()
            os.chdir(self.sandbox.name)
            self.runner = CommandRunner(
                command=self.get_execute_command(self.sandbox),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                print_command=False,
                interpreter=self.interpreter,
            )
            self.runner.run()
            os.chdir(cwd)

            stdout_check = self.runner.check_stdout(self.exp_stdout_path)
            stderr_check = self.runner.check_stderr(self.exp_stderr_path)
            status = True
            if self.exp_exit_status is not None:
                status = self.exp_exit_status == self.runner.exit_status
            # TODO self.result.exit_status
            self.exit_status: int = self.runner.exit_status
            self.result.passed = stdout_check and stderr_check and status
            self.result.score = self.point_value if self.result.passed else 0

            # TODO report to students
            print(
                f"stdout check: {stdout_check}; stderr check: {stderr_check}; status: {status}"
            )
        except subprocess.TimeoutExpired:
            logger.info(f"{self.name} timed out (limit={self.timeout}s)!")
            self.result.timed_out = True


class RealtimeTestCaseBulkLoader:
    def __init__(self, autograder, generator: str, prefix: str = "", default_timeout: float = 10, interpreter: str | None = None):
        self.autograder = autograder
        self.generator: str = generator
        self.sandbox: TemporaryDirectory = autograder.sandbox
        self.sandbox_reference: TemporaryDirectory = autograder.sandbox_reference
        self.default_timeout: float = default_timeout
        self.prefix: str = prefix
        self.interpreter: str | None = interpreter

    def add(
        self,
        name: str,
        id: int = -1,
        point_value: float = 1,
        hidden: bool = False,
        prefix: str = "",
        timeout: float | None = None,
        hidden_msg: str = "hidden test",
    ):
        if timeout is None:
            timeout = self.default_timeout
        test_case = RealtimeTestCase(
            generator=self.generator,
            sandbox=self.sandbox,
            sandbox_reference=self.sandbox_reference,
            id=id,
            name=prefix + name,
            point_value=point_value,
            timeout=timeout,
            interpreter=self.interpreter,
            hidden=hidden,
            hidden_msg=hidden_msg,
        )

        self.autograder.add_test(test_case)
        return self

    def add_list(
        self,
        test_list: List[Tuple[str, float]],
        prefix: str = "",
        hidden: bool = False,
        timeout: float | None = None,
        hidden_msg: str = "hidden test",
    ):
        for i, (name, point_value) in enumerate(test_list):
            self.add(
                name=name,
                id=i+1,
                point_value=point_value,
                hidden=hidden,
                timeout=timeout,
                prefix=prefix,
                hidden_msg=hidden_msg,
            )
        return self
