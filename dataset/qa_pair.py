from typing import List


class QAPair:

    def __init__(self, task: str):
        self.task = task

    def from_row(self, row):
        self.gif_name: str = row.gif_name
        self.question: str = row.question.lower()

        if self.task in ['action', 'trans']:
            self.answers: List[str] = [
                row.a0, row.a1, row.a2, row.a3, row.a4
            ]
            self.answer_index: int = row.answer

        else:
            self.answer_index: str = row.answer
