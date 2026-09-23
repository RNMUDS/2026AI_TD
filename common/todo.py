"""ノートブックの TODO の書き忘れを見つける．

    todo_check("TODO 1")

いま実行しているセルに「...」（ここに書く，という目印）が残っていたら，日本語のメッセージで止める．
コメント（# 以降）と文字列の中の「...」は数えない．
"""
import io
import tokenize


def _has_placeholder(src):
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.OP and tok.string == "...":
                return True
    except (tokenize.TokenError, IndentationError):
        pass
    return False


def todo_check(name):
    try:
        from IPython import get_ipython
        src = get_ipython().history_manager.input_hist_raw[-1]   # いま実行しているセル
    except Exception:
        return   # ノートブックの外では何もしない
    if _has_placeholder(src):
        raise RuntimeError(f"{name} が未記入です：「...」を消して式を書いてください")
