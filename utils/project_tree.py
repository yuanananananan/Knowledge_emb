from pathlib import Path

def tree(path: Path, prefix: str = ''):
    """递归打印目录树，类似 Linux tree 命令"""
    pointers = ['├── ', '└── ']
    contents = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
    for idx, item in enumerate(contents):
        pointer = pointers[-1] if idx == len(contents) - 1 else pointers[0]
        yield prefix + pointer + item.name
        if item.is_dir():
            extension  = '│   ' if idx != len(contents) - 1 else '    '
            yield from tree(item, prefix + extension)

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    print(root.name)
    for line in tree(root):
        print(line)