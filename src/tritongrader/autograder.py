import os
import logging
import shutil
import platform

from tempfile import TemporaryDirectory
from typing import List, Optional
from tritongrader.runner import CommandRunner
import subprocess

from tritongrader.test_case import (
    TestCaseBase,
    IOTestCaseBulkLoader,
    BasicTestCase,
    CustomTestCase,
    CustomTestResult,
    RealtimeTestCaseBulkLoader,
)

logger = logging.getLogger("tritongrader.autograder")


class Autograder:
    """
    An autograder object defines a single set of tests that can be applied to
    parts of an assignment that share a common set of source files and build
    procedure (e.g. Makefile).
    """

    def __init__(
        self,
        name: str,
        submission_path: str,
        tests_path: str,
        working_directory: str = "",
        required_files: List[str] = [],
        supplied_files: List[str] = [],
        verbose_rubric: bool = False,
        build_command: Optional[str] = None,
        compile_points: int = 0,
        missing_files_check: bool = True,
        interpreter: Optional[str] = None,
        reference_path: str | None = None,
        reference_build_command: str | None = None,
    ):
        """
        Note: `build_command` must be given for compilation to happen; there is not implicit build
        command.
        """

        self.name = name

        self.tests_path = tests_path
        self.submission_path = submission_path
        self.working_directory = working_directory

        self.interpreter = interpreter
        self.required_files = required_files
        self.supplied_files = supplied_files
        self.verbose_rubric = verbose_rubric
        self.compile_points = compile_points

        self.reference_path = reference_path
        self.reference_build_command = reference_build_command

        self.test_cases: List[TestCaseBase] = []

        # A sandbox directory where submission and test files will be copied to.
        self.sandbox: TemporaryDirectory = self.create_sandbox_directory()
        self.sandbox_reference: TemporaryDirectory = self.create_sandbox_directory()

        self.missing_files_check_test_case = None
        if missing_files_check:
            self.missing_files_check_test_case: CustomTestCase = (
                self.create_missing_files_check_test_case(required_files)
            )
            self.add_test(self.missing_files_check_test_case)

        self.build_test_case = None
        if build_command:
            self.build_test_case: BasicTestCase = self.create_build_test_case(
                build_command, compile_points
            )
            self.add_test(self.build_test_case)

    def create_missing_files_check_test_case(
        self, required_files: List[str]
    ) -> CustomTestCase:
        def check_missing_files(result: CustomTestResult):
            logger.info("Checking missing files...")
            missing_files = []
            for filename in required_files:
                fpath = os.path.join(self.submission_path, filename)
                if not os.path.exists(fpath):
                    missing_files.append(filename)
            if not missing_files:
                result.output = "All required files have been located."
                result.passed = True
            else:
                result.output = "\n".join(["Missing files"] + missing_files)
                result.passed = False

        return CustomTestCase(
            check_missing_files, name="Missing Files Check", point_value=0
        )

    def create_build_test_case(self, build_command, point_value=0) -> BasicTestCase:
        return BasicTestCase(
            command=build_command,
            name="Compiling",
            point_value=point_value,
            expected_retcode=0,
            timeout=3,
            interpreter=self.interpreter,
        )

    def create_sandbox_directory(self) -> TemporaryDirectory:
        tmpdir = TemporaryDirectory(prefix="Autograder_")
        logger.info(f"Sandbox created at {tmpdir.name}")
        return tmpdir

    def add_test(self, test_case: TestCaseBase):
        """
        Add a test case of any kind to the autograder.
        """
        test_case.name = f"{self.name}: {test_case.name}"
        self.test_cases.append(test_case)

    def io_tests_bulk_loader(
        self,
        prefix: str = "",
        default_timeout: float = 1,
        binary_io: bool = False,
        commands_path: Optional[str] = None,
        test_input_path: Optional[str] = None,
        expected_stdout_path: Optional[str] = None,
        expected_stderr_path: Optional[str] = None,
        expected_exit_status_path: Optional[str] = None,
        commands_prefix: Optional[str] = "cmd-",
        test_input_prefix: Optional[str] = "in-",
        expected_stdout_prefix: Optional[str] = "out-",
        expected_stderr_prefix: Optional[str] = "err-",
    ) -> IOTestCaseBulkLoader:
        """
        Creates a bulk loader for I/O-based test cases to create tests
        in batches with settings configured by the bulk loader.

        Two chainable methods are available in the bulk loader: .add()
        and .add_list(). The methods can be chained like so:

        ```
        ag.io_test_bulk_loader(...).add(...).add(...).add_list(...)
        ```

        with the desired parameters for the bulk loader and the add methods.
        """
        return IOTestCaseBulkLoader(
            self,
            commands_path=(commands_path or os.path.join(self.tests_path, "in")),
            test_input_path=(test_input_path or os.path.join(self.tests_path, "in")),
            expected_stdout_path=(
                expected_stdout_path or os.path.join(self.tests_path, "exp")
            ),
            expected_stderr_path=(
                expected_stderr_path or os.path.join(self.tests_path, "exp")
            ),
            expected_exit_status_path=(
                expected_exit_status_path or os.path.join(self.tests_path, "exp")
            ),
            commands_prefix=commands_prefix,
            test_input_prefix=test_input_prefix,
            expected_stdout_prefix=expected_stdout_prefix,
            expected_stderr_prefix=expected_stderr_prefix,
            prefix=prefix,
            default_timeout=default_timeout,
            binary_io=binary_io,
        )

    def realtime_tests_bulk_loader(
        self,
        generator: str,
        prefix: str = "",
        binary_io: bool = False,
        default_timeout: float = 3,
    ) -> RealtimeTestCaseBulkLoader:
        return RealtimeTestCaseBulkLoader(
            autograder=self,
            generator=generator,
            prefix=prefix,
            binary_io=binary_io,
            default_timeout=default_timeout,
            interpreter=self.interpreter
        )

    def copy2sandbox(self, sandbox, src_dir, item):
        path = os.path.realpath(os.path.join(src_dir, item))
        dst = os.path.join(sandbox.name, item)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(path):
            shutil.copy2(path, dst)
            logger.info(f"Copied file from {path} to {dst}...")
        elif os.path.isdir(path):
            shutil.copytree(path, dst)
            logger.info(f"Copied directory from {path} to {dst}...")

    def copy_submission_files(self):
        for f in self.required_files:
            self.copy2sandbox(self.sandbox, self.submission_path, f)

    def copy_supplied_files(self):
        for f in self.supplied_files:
            self.copy2sandbox(self.sandbox, self.tests_path, f)

    def _execute(self):
        self.copy_submission_files()
        self.copy_supplied_files()

        for test in self.test_cases:
            test.execute()

            if test == self.missing_files_check_test_case and not test.result.passed:
                logger.info("Some files are missing. Aborting autograder.")
                break

            if test == self.build_test_case and not test.result.passed:
                logger.info("Failed to compile. Aborting autograder.")
                break

            if test.early_stop and not test.result.passed:
                logger.info("Early stop. Aborting autograder.")
                break

    def _execute_reference(self):
        if not self.reference_path or not self.reference_build_command:
            return

        os.makedirs(os.path.join(self.sandbox_reference.name, self.working_directory), exist_ok=True)
        for f in os.listdir(self.reference_path):
            self.copy2sandbox(self.sandbox_reference, self.reference_path, f)

        try:
            runner = CommandRunner(
                command=self.reference_build_command,
                capture_output=True,
                timeout=10,
            )
            runner.run()
            if runner.exit_status != 0:
                logger.info(f"{self.name} reference build failed!")
                raise Exception(f"{self.name} reference build failed!")
        except subprocess.TimeoutExpired:
            logger.info(f"{self.name} reference build time out!")
            raise Exception(f"{self.name} reference build time out!")


    def execute(self):
        logger.debug(platform.uname())

        if self.reference_path and self.reference_build_command:
            logger.info(f"Building {self.name} reference in {self.sandbox_reference.name}...")
            os.makedirs(os.path.join(self.sandbox_reference.name, self.working_directory), exist_ok=True)
            cwd = os.getcwd()
            os.chdir(os.path.join(self.sandbox_reference.name, self.working_directory))
            self._execute_reference()
            logger.info(f"Finished building {self.name} reference. Returning to {cwd}")
            os.chdir(cwd)

        logger.info(f"Running {self.name} test(s) in {self.sandbox.name}...")
        os.makedirs(os.path.join(self.sandbox.name, self.working_directory), exist_ok=True)
        cwd = os.getcwd()
        os.chdir(os.path.join(self.sandbox.name, self.working_directory))
        self._execute()
        logger.info(f"Finished running {self.name} test(s). Returning to {cwd}")
        os.chdir(cwd)

