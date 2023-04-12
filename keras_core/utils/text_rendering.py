import shutil


class Table:
    def __init__(self, header, rows, positions, alignments=None, max_line_length=80):
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
        line_length = min(max_line_length, shutil.get_terminal_size().columns - 4)
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

    def print_row(
        self,
        fields,
        vertical_separator="│",
        alignments=None,
    ):
        alignments = alignments or ["center" for _ in fields]
        lines = []
        line_break_conditions = ("),", "},", "],", "',", ") ")
        for field, width in zip(fields, self.column_widths):
            buffered_width = width - 1
            if len(field) < buffered_width and not "\n" in field:
                lines.append([field])
                continue
            subfields = []
            while len(field) >= buffered_width or "\n" in field:
                if "\n" in field[:buffered_width]:
                    # priority: break on line break
                    cutoff = field.find("\n")
                    subfield = field[:cutoff]
                    field = field[cutoff + 1:]
                    subfields.append(subfield)
                    continue
                # secondary: break on certain characters
                candidate_cutoffs = [
                    field.find(x) + len(x)
                    for x in line_break_conditions
                    if 0 < field.find(x) < buffered_width
                ]
                if candidate_cutoffs:
                    cutoff = min(buffered_width - 1, *candidate_cutoffs)
                else:
                    cutoff = buffered_width - 1
                subfield = field[:cutoff]
                field = field[cutoff:]
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
            line = vertical_separator.join(self.align_field(field, width, alignment) for field, width, alignment in zip(fields, self.column_widths, alignments))
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
        lines.append(self.print_row(self.header, vertical_separator="┃"))
        lines.append(self.make_separator(*"┡╇┩━"))

        # Print rows
        for i, row in enumerate(self.rows):
            lines.append(self.print_row(row, alignments=self.alignments))
            if i < len(self.rows) - 1:
                lines.append(self.make_separator(*"├┼┤─"))

        lines.append(self.make_separator(*"└┴┘─"))
        return "\n".join(lines)
