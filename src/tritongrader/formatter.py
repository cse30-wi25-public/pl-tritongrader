import os
import json
import logging
from typing import Dict, Callable, List, Union, Iterable, Optional
from difflib import HtmlDiff
from tritongrader import Autograder

from tritongrader.test_case import TestCaseBase
from tritongrader.test_case import IOTestCase
from tritongrader.test_case import BasicTestCase
from tritongrader.test_case import CustomTestCase

logger = logging.getLogger("tritongrader.formatter")


class ResultsFormatterBase:
    def __init__(self, src: Union[Autograder, Iterable[Autograder]]):
        self.formatters: Dict[TestCaseBase, Callable[[TestCaseBase], None]] = {
            IOTestCase: self.format_io_test,
            BasicTestCase: self.format_basic_test,
            CustomTestCase: self.format_custom_test,
        }
        self.test_cases: List[TestCaseBase] = []
        ags = [src] if isinstance(src, Autograder) else src
        for autograder in ags:
            self.test_cases.extend(autograder.test_cases)

    def format_io_test(self, test: IOTestCase):
        raise NotImplementedError

    def format_basic_test(self, test: BasicTestCase):
        raise NotImplementedError

    def format_custom_test(self, test: CustomTestCase):
        raise NotImplementedError

    def format_test(self, test: TestCaseBase):
        return self.formatters[type(test)](test)

    def execute(self):
        raise NotImplementedError


class GradescopeResultsFormatter(ResultsFormatterBase):
    def __init__(
        self,
        src: Union[Autograder, Iterable[Autograder]],
        message: str = "",
        visibility: str = "visible",
        stdout_visibility: str = "hidden",
        hidden_tests_setting: str = "hidden",
        hide_points: bool = False,
        max_output_bytes: int = 5000,
        verbose: bool = True,
        html_diff: bool = False,
    ):
        super().__init__(src)
        self.message = message
        self.visibility: str = visibility
        self.stdout_visibility: str = stdout_visibility
        self.hidden_tests_setting: str = hidden_tests_setting
        self.hide_points: bool = hide_points
        self.max_output_bytes: int = max_output_bytes
        self.verbose: bool = verbose
        self.html_diff: bool = html_diff
        self.results: dict = None

    def html_diff_make_table(
        self,
        fromtext: str,
        totext: str,
        fromdesc: str = "",
        todesc: str = "",
    ):
        return HtmlDiff(tabsize=2, wrapcolumn=80).make_table(
            fromlines=fromtext.split("\n"),
            tolines=totext.split("\n"),
            fromdesc=fromdesc,
            todesc=todesc,
            context=True,
            numlines=3,
        )

    def generate_html_diff(self, test: IOTestCase):
        stdout_diff = self.html_diff_make_table(
            fromtext=test.actual_stdout or "",
            totext=test.expected_stdout or "",
            fromdesc="Your stdout",
            todesc="Expected stdout",
        )
        stderr_diff = self.html_diff_make_table(
            fromtext=test.actual_stderr or "",
            totext=test.expected_stderr or "",
            fromdesc="Your stderr",
            todesc="Expected stderr",
        )
        html = "".join(
            [
                "<div>",
                "<h2>exit status</h2>",
                str(test.runner.exit_status),
                "<hr>",
                "<h2>stdout</h2>",
                stdout_diff,
                "<hr>",
                "<h2>stderr</h2>",
                stderr_diff,
                "</div>",
            ]
        )
        return html

    def basic_io_output(self, test: IOTestCase):
        if not test.result.has_run or not test.runner:
            return "This test was not run."

        if test.result.error:
            # TODO report to Observer
            return "\n".join(
                [
                    "=== Unexpected autograder runtime error!  Please notify your instructors. ===",
                    "=== stdout ===",
                    self.cutter(test.actual_stdout),
                    "=== stderr ===",
                    self.cutter(test.actual_stderr),
                ]
            )
        if test.result.timed_out:
            return "\n".join(
                [
                    f"Test case timed out with limit = {test.timeout}.",
                    "== stdout ==",
                    self.cutter(test.actual_stdout),
                    "== stderr ==",
                    self.cutter(test.actual_stderr),
                ]
            )

        status_str = "PASSED" if test.result.passed else "FAILED"
        summary = []
        summary.append(f"{status_str} in {test.runner.running_time:.2f} ms.")

        if self.verbose:
            summary.extend(["=== test command ===", test.command])

            if test.test_input is not None:
                summary.extend(["=== test input ===", test.test_input])
            summary.extend(
                [
                    "=== expected stdout ===",
                    self.cutter(test.expected_stdout),
                    "=== expected stderr ===",
                    self.cutter(test.expected_stderr),
                    "=== expected exit status ===",
                    self.cutter(str(test.exp_exit_status)),
                ]
            )
            if not test.result.passed:
                summary.extend(
                    [
                        "=== your stdout ===",
                        self.cutter(test.actual_stdout),
                        "=== your stderr ===",
                        self.cutter(test.actual_stderr),
                        "=== your exit status ===",
                        self.cutter(str(test.exit_status)),
                    ]
                )

        return "\n".join(summary)

    def format_io_test(self, test: IOTestCase):
        return {
            "output_format": "html" if self.html_diff else "simple_format",
            "output": (self.generate_html_diff(test) if self.html_diff else self.basic_io_output(test)),
        }

    def format_basic_test(self, test: BasicTestCase):
        if not test.runner:
            return {"output": "This test was not run."}
        summary = []
        summary.extend(
            [
                "=== test command ===",
                self.cutter(test.command),
                "=== exit status ===",
                self.cutter(str(test.runner.exit_status)),
            ]
        )
        if self.verbose:
            summary.extend(
                [
                    "=== stdout ===",
                    self.cutter(test.runner.stdout),
                    "=== stderr ===",
                    self.cutter(test.runner.stderr),
                ]
            )
        return {"output": "\n".join(summary)}

    def format_custom_test(self, test: CustomTestCase):
        if not test.result.has_run:
            output = "This test was not run."
        else:
            output = self.cutter(test.result.output)

        return {"output": output}

    def format_test(self, test: TestCaseBase):
        item = {
            "name": test.name,
        }
        item["points"] = test.result.score
        if test.point_value is not None:
            item["max_points"] = test.point_value

        item.update(super().format_test(test))
        return item

    def get_total_score(self):
        return sum(i.result.score for i in self.test_cases)

    def get_full_score(self):
        return sum(i.point_value for i in self.test_cases if i.point_value is not None)

    def cutter(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        encoding = "utf-8"
        placeholder = "...(omitted {count} lines)..."
        raw_bytes = text.encode(encoding)
        if len(raw_bytes) <= self.limitsize:
            return text

        lines = text.splitlines()

        def line_size(line: str) -> int:
            return len(line.encode(encoding)) + 1

        head_lines = []
        head_acc_bytes = 0
        for ln in lines:
            size_ln = line_size(ln)
            if head_acc_bytes + size_ln <= self.headsize:
                head_lines.append(ln)
                head_acc_bytes += size_ln
            else:
                break

        tail_lines = []
        tail_acc_bytes = 0
        for ln in reversed(lines):
            size_ln = line_size(ln)
            if tail_acc_bytes + size_ln <= self.tailsize:
                tail_lines.append(ln)
                tail_acc_bytes += size_ln
            else:
                break
        tail_lines.reverse()

        omitted_lines_count = len(lines) - len(head_lines) - len(tail_lines)
        if omitted_lines_count > 0:
            placeholder_line = placeholder.format(count=omitted_lines_count)
        else:
            placeholder_line = ""

        if placeholder_line:
            truncated_parts = head_lines + [placeholder_line] + tail_lines
        else:
            truncated_parts = head_lines
            truncated_parts += tail_lines[len(head_lines) :]

        final_text = "\n".join(part for part in truncated_parts if part != "")
        final_bytes = final_text.encode(encoding)

        if len(final_bytes) > self.limitsize:
            final_text = "Text is too large to display.\n"

        final_text = "[WARN]: Text too long, truncated.\n" + final_text
        return final_text

    def execute(self):
        logger.info("Formatter running...")
        self.results = {
            "gradable": True,
            "score": self.get_total_score() / self.get_full_score(),
            "message": self.message,
            "tests": [self.format_test(i) for i in self.test_cases],
        }

        if self.hide_points:
            self.results["score"] = 0
        logger.info("Formatter execution completed.")
        return self.results

    def export(self, path="/autograder/results/results.json", limit: int = 2**30):
        # NOTE: need better estimation
        num_tests = sum(1 for _ in self.test_cases)
        estimated_size = (limit - 10000) // num_tests // 3
        self.limitsize = estimated_size
        self.headsize = int(estimated_size * 0.1)
        self.tailsize = int(estimated_size * 0.1)

        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, "w+") as fp:
            json.dump(self.execute(), fp)


if __name__ == "__main__":
    formatter = GradescopeResultsFormatter()
    formatter.formatters[IOTestCase](None)
