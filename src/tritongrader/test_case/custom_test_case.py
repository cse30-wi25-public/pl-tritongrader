import logging

from typing import Callable

from tritongrader.test_case.test_case_base import TestCaseBase, TestResultBase

logger = logging.getLogger("tritongrader.test_case")


class CustomTestResult(TestResultBase):
    pass


class CustomTestCase(TestCaseBase):
    def __init__(
        self,
        func: Callable[[CustomTestResult], None],
        name: str = "Test Case",
        point_value: float = 1,
        timeout: float = TestCaseBase.DEFAULT_TIMEOUT,
        hidden: bool = False,
        early_stop: bool = False,
    ):
        super().__init__(name, point_value, timeout, hidden)
        self.test_func: Callable[[CustomTestResult], None] = func
        self.result: CustomTestResult = CustomTestResult()
        self.early_stop = early_stop

    def execute(self):
        self.result.has_run = True
        self.test_func(self.result)
