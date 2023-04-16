import os


def count_loc(directory, exclude=("_test",), extensions=(".py",), verbose=0):
    loc = 0
    for root, _, fnames in os.walk(directory):
        skip = False
        for ex in exclude:
            if root.endswith(ex):
                skip = True
        if skip:
            continue

        for fname in fnames:
            skip = False
            for ext in extensions:
                if not fname.endswith(ext):
                    skip = True
                    break

                for ex in exclude:
                    if fname.endswith(ex + ext):
                        skip = True
                        break
            if skip:
                continue

            fname = os.path.join(root, fname)
            if verbose:
                print(f"Count LoCs in {fname}")

            with open(fname) as f:
                lines = f.read().split("\n")

            string_open = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not string_open:
                    if not line.startswith('"""'):
                        loc += 1
                    else:
                        if not line.endswith('"""'):
                            string_open = True
                else:
                    if line.startswith('"""'):
                        string_open = False
    return loc
