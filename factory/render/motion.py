"""Velocity-limited reframing, with continuous local effects for spoken asides."""
import math


def choose_motion(role, index, duration=4):
    if duration < 1.5:
        return "none"
    role = str(role or "")
    # Stronger push on story turns; light push elsewhere so shots feel alive.
    if role in ("twist", "payoff", "爆点", "转折"):
        return "push_strong"
    if role in ("hook", "escalation"):
        return "push_in"
    if duration >= 2.5 and index % 5 == 0:
        return "pan_right" if index % 10 == 0 else "pan_left"
    return "push_in"


def motion_amount(duration, returning=False, kind="push_in"):
    """Bound peak zoom speed even when the requested shot is very short."""
    if returning:
        return min(.025, max(0, float(duration)) * .0035)
    strong = kind == "push_strong"
    # Light default stays at or below ~2% peak; turns get a bit more.
    cap = .032 if strong else .02
    rate = .0065 if strong else .005
    return min(cap, max(0, float(duration)) * rate)


def motion_sample(kind, time, duration, returning=False):
    """Numerical counterpart used to check smoothness independently of ffmpeg."""
    u = max(0., min(1., float(time) / max(.001, duration)))
    ease = math.sin(math.pi * u) ** 2 if returning else 3 * u * u - 2 * u * u * u
    zoom = 1 + motion_amount(duration, returning, kind) * ease
    fraction = .15 + .7 * (3*u*u-2*u*u*u) if kind == "pan_right" else .85-.7*(3*u*u-2*u*u*u)
    return zoom, fraction if kind.startswith("pan_") else .5


def motion_filter(kind, width, height, duration, frames, oversample=3, local_spans=None):
    if not local_spans and kind not in ("push_in", "push_strong", "pan_left", "pan_right"):
        return "null"
    if frames <= 1:
        return "null"
    factor = max(2, min(4, int(oversample)))
    sw, sh = int(width) * factor, int(height) * factor
    if local_spans:
        zoom_terms, shifts = [], []
        for span in local_spans:
            a, b = int(span["start_frame"]), int(span["end_frame"]) - 1
            mode = span.get("motion", "push_in")
            if mode == "none" or b <= a:
                continue
            u = "(on-%d)/%d" % (a, b - a)
            delta = motion_amount((b - a) / 24, returning=True, kind=mode)
            pulse = "if(between(on,%d,%d),%.9f*pow(sin(PI*(%s)),2),0)" % (a, b, delta, u)
            zoom_terms.append(pulse)
            if mode.startswith("pan_"):
                sign = 1 if mode == "pan_right" else -1
                shifts.append("if(between(on,%d,%d),%d*0.35*(2*(%s)-1),0)" % (a, b, sign, u))
        if not zoom_terms:
            return "null"
        zoom = "1+" + "+".join(zoom_terms)
        fraction = ".5" + "".join("+(%s)" % term for term in shifts)
        x = "(iw-iw/zoom)*(%s)" % fraction
    else:
        u = "min(on/%d,1)" % max(1, frames - 1)
        ease = "(3*pow(%s,2)-2*pow(%s,3))" % (u, u)
        zoom = "1+%.9f*%s" % (motion_amount(duration, kind=kind), ease)
        frac = "(.15+.7*%s)" % ease if kind == "pan_right" else "(.85-.7*%s)" % ease
        x = "(iw-iw/zoom)*%s" % frac if kind.startswith("pan_") else "(iw-iw/zoom)/2"
    # Two-times output plus Lanczos keeps subpixel motion smooth. Unlike the
    # old three/four-times output this does not retain enormous full HD frames.
    ow, oh = int(width) * 2, int(height) * 2
    return ("format=yuv444p,scale=%d:%d:flags=lanczos,"
            "zoompan=z='%s':x='%s':y='(ih-ih/zoom)/2':d=1:s=%dx%d:fps=24,"
            "scale=%d:%d:flags=lanczos,setsar=1,format=yuv420p") % (sw, sh, zoom, x, ow, oh, width, height)
