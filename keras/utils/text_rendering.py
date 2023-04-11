# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import shutil


class TextTable:
    def __init__(
        self, header, rows, positions, alignments=None, max_line_length=80
    ):
        if len(header) != len(positions):
            raise ValueError("header and positions should be the same length.")
        if not all(p <= 1.0 for p in positions):
            raise ValueError("All positions should be <= 1.")
        self.alignments = alignments or ["center" for _ in header]
        if len(self.alignments) != len(header):
            raise ValueError("header and alignments should be the same length.")
        last_p = 0.0
        for p in positions:
            if p <= last_p:
                raise ValueError(
                    "All consecutive positions should be greater than the last."
                )
            last_p = p
        self.header = header
        self.rows = rows

        # Compute columns widths
        line_length = min(
            max_line_length, shutil.get_terminal_size().columns - 4
        )
        column_widths = []
        current = 0
        for pos in positions:
            width = int(pos * line_length) - current
            if width < 4:
                raise ValueError("Insufficient console width to print summary.")
            column_widths.append(width)
            current += width
        self.column_widths = column_widths

    def make_separator(self, left, mid, right, horizontal):
        line = mid.join(horizontal * width for width in self.column_widths)
        return f"{left}{line}{right}"

    @staticmethod
    def maybe_pad(field, alignment):
        if alignment == "left":
            return " " + field
        if alignment == "right":
            return field + " "
        return field

    def print_row(
        self,
        fields,
        vertical_separator="│",
        alignments=None,
        highlight=False,
    ):
        alignments = alignments or ["center" for _ in fields]
        lines = []
        line_break_chars_post = (")", "}", "]")
        line_break_chars_pre = ("(", "{", "[")
        for field, width, alignment in zip(
            fields, self.column_widths, alignments
        ):
            field = self.maybe_pad(str(field), alignment)
            buffered_width = width - 1
            if len(field) < buffered_width and "\n" not in field:
                lines.append([field])
                continue
            subfields = []
            while len(field) >= buffered_width or "\n" in field:
                if "\n" in field[:buffered_width]:
                    # priority: break on line break
                    cutoff = field.find("\n")
                    subfield = field[:cutoff]
                    field = field[cutoff + 1 :]
                    field = self.maybe_pad(field, alignment)
                    subfields.append(subfield)
                    continue
                # secondary: break on certain characters
                candidate_cutoffs_post = [
                    field.find(x) + len(x)
                    for x in line_break_chars_post
                    if 0 < field.find(x) < buffered_width - len(x)
                ]
                candidate_cutoffs_pre = [
                    field.find(x)
                    for x in line_break_chars_pre
                    if 0 < field.find(x) < buffered_width
                ]
                cutoffs = candidate_cutoffs_post + candidate_cutoffs_pre
                if cutoffs:
                    cutoff = max(cutoffs)
                else:
                    cutoff = buffered_width - 1
                subfield = field[:cutoff]
                field = field[cutoff:]
                field = self.maybe_pad(field, alignment)
                subfields.append(subfield)
            if field:
                subfields.append(field)
            lines.append(subfields)

        max_subfield_count = max(len(subs) for subs in lines)
        rendered_lines = []
        for i in range(max_subfield_count):
            fields = []
            for subfields in lines:
                if len(subfields) < i + 1:
                    field = ""
                else:
                    field = subfields[i]
                fields.append(field)
            aligned_fields = [
                self.align_field(field, width, alignment)
                for field, width, alignment in zip(
                    fields, self.column_widths, alignments
                )
            ]
            if highlight:
                aligned_fields = [
                    highlight_msg(field) for field in aligned_fields
                ]
            line = vertical_separator.join(aligned_fields)
            line = f"{vertical_separator}{line}{vertical_separator}"
            rendered_lines.append(line)
        return "\n".join(rendered_lines)

    @staticmethod
    def align_field(field, width, alignment):
        if alignment == "center":
            return field.center(width)
        if alignment == "left":
            return field.ljust(width)
        if alignment == "right":
            return field.rjust(width)

    def make(self):
        lines = []
        # Print header
        lines.append(self.make_separator(*"┏┳┓━"))
        lines.append(
            self.print_row(self.header, vertical_separator="┃", highlight=True)
        )
        lines.append(self.make_separator(*"┡╇┩━"))

        # Print rows
        for i, row in enumerate(self.rows):
            lines.append(self.print_row(row, alignments=self.alignments))
            if i < len(self.rows) - 1:
                lines.append(self.make_separator(*"├┼┤─"))

        lines.append(self.make_separator(*"└┴┘─"))
        return "\n".join(lines)


def highlight_msg(msg):
    return f"\x1b[1m{msg}\x1b[0m"
