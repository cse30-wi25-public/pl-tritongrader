class TestResultBase:
    def __init__(self):
        self.score: int = 0
        self.passed: bool = False
        self.timed_out: bool = False
        self.error: bool = False
        self.running_time: float = None
        self.has_run: bool = False


class TestCaseBase:
    DEFAULT_TIMEOUT = 10

    def __init__(
        self,
        name: str = "Test Case",
        point_value: float = 1,
        timeout: float = DEFAULT_TIMEOUT,
        hidden: bool = False,
        hidden_msg: str = "hidden test",
        early_stop: bool = False
    ):
        self.name: str = name
        self.point_value: float = point_value
        self.timeout: float = timeout
        self.hidden: bool = hidden
        self.result: TestResultBase = None
        self.hidden_msg: str = hidden_msg
        self.early_stop: bool = early_stop

    def execute(self) -> TestResultBase:
        raise NotImplementedError
