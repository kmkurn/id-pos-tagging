import typing

from camel import PYTHON_TYPES, Camel, CamelRegistry

from models import MemorizationTagger

registry = CamelRegistry()


@registry.dumper(MemorizationTagger, 'memo', version=1)
def _dump_memo_v1(memo: MemorizationTagger) -> dict:
    return {
        'mapping': memo.mapping,
        'window': memo.window,
    }


@registry.loader('memo', version=1)
def _load_memo_v1(data: dict, version: int) -> MemorizationTagger:
    return MemorizationTagger(data['mapping'], window=data['window'])


def dump(obj: typing.Any) -> str:
    return Camel([registry, PYTHON_TYPES]).dump(obj)


def load(data: str) -> typing.Any:
    return Camel([registry]).load(data)
