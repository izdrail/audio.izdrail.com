from fastrtc import Stream, ReplyOnPause
def h(a, b): pass
s = Stream(handler=ReplyOnPause(h), modality='audio')
print(f"Attributes: {dir(s)}")
if hasattr(s, 'handlers'):
    print(f"Handlers: {s.handlers}")
    print(f"Handlers type: {type(s.handlers)}")
if hasattr(s, 'handler'):
    print(f"Handler: {s.handler}")
