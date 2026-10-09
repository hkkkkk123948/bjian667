"""A small portable DTW backend for Whisper word clocks on Windows.

Numba's Windows LLVM runtime can fail while finalising compiled DTW code.
Narration is short enough to use the same dynamic-programming calculation in
NumPy/Python without compiling native code or changing the speech model.
"""


def numpy_dtw(distances):
    import numpy as np

    matrix = np.asarray(distances)
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ValueError("词时间对齐需要非空二维距离矩阵")
    rows, columns = matrix.shape
    costs = np.full((rows+1, columns+1), np.inf, dtype=np.float32)
    choices = np.zeros((rows+1, columns+1), dtype=np.uint8)
    costs[0, 0] = 0
    for row in range(1, rows+1):
        for column in range(1, columns+1):
            diagonal = costs[row-1, column-1]
            vertical = costs[row-1, column]
            horizontal = costs[row, column-1]
            # Whisper uses a horizontal move when costs tie. Keeping that
            # rule makes the resulting word clock reproducible.
            if diagonal < vertical and diagonal < horizontal:
                best, move = diagonal, 0
            elif vertical < diagonal and vertical < horizontal:
                best, move = vertical, 1
            else:
                best, move = horizontal, 2
            costs[row, column] = matrix[row-1, column-1] + best
            choices[row, column] = move
    row, column = rows, columns
    path = []
    while row or column:
        path.append((row-1, column-1))
        move = choices[row, column] if row and column else (1 if row else 2)
        row -= move != 2
        column -= move != 1
    return np.asarray(path[::-1], dtype=np.int64).T
