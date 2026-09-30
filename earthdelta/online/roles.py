"""Frozen date roles; target rows are distinct from available feedback history."""
from .timeindex import index_of, checked_index


def role(index):
    i=checked_index(index)
    if i%4: raise ValueError('target issue must be daily 00Z')
    for name,start,end in (
        ('estimate','2020-01-01','2020-02-24'),
        ('gap','2020-02-25','2020-02-29'),
        ('select','2020-03-01','2020-03-26'),
        ('warmup','2020-03-27','2020-03-31'),
        ('analysis','2020-04-01','2020-12-26')):
        if index_of(start+'T00:00:00Z')<=i<=index_of(end+'T00:00:00Z'):return name
    raise ValueError('date outside registered target roster')


def require_fit_rows(indices,*,final_refit=False):
    values=list(indices)
    if not values or len(values)!=len(set(values)):
        raise ValueError('unique nonempty training rows required')
    allowed={'estimate','select'} if final_refit else {'estimate'}
    if any(role(i) not in allowed for i in values):
        raise PermissionError('forbidden target role in fit')
    return tuple(values)
