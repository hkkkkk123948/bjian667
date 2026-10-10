"""Read-only render budgeting; never close other apps or change OS settings."""
import os


def available_memory():
    try:
        if os.name == 'nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                            ('physical',ctypes.c_ulonglong),('available_physical',ctypes.c_ulonglong),
                            ('page_file',ctypes.c_ulonglong),('available_page_file',ctypes.c_ulonglong),
                            ('virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),
                            ('extended',ctypes.c_ulonglong)]
            status=MemoryStatus()
            status.length=ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return min(status.available_physical,status.available_page_file)
        return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    except (AttributeError,OSError,ValueError):
        return None


def render_budget(params):
    """Memory budget for the scale/crop fallback.

    The cubic warp path does not read motion_oversample. When the fallback
    runs, its supersample never drops below 8, even if this budget lowers
    the requested value on a small machine.
    """
    result=dict(params)
    free=available_memory()
    requested=int(result.get('motion_oversample',3))
    if (free is not None and free < 2*1024**3 and
            int(result.get('target_width',1920))*int(result.get('target_height',1080)) >= 1280*720):
        result['motion_oversample']=min(requested,2)
    return result,{'available_memory_bytes':free,'requested_motion_oversample':requested,
                   'effective_motion_oversample':int(result.get('motion_oversample',3)),
                   'encoder_threads':1,'encoder_lookahead_frames':0,'resolution_preserved':True}
