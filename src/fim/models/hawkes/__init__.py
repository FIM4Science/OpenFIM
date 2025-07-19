from .hawkes import FIMHawkes, FIMHawkesConfig


# Register with transformers for proper AutoConfig/AutoModel support
FIMHawkesConfig.register_for_auto_class()
FIMHawkes.register_for_auto_class("AutoModel")

__all__ = ["FIMHawkes", "FIMHawkesConfig"]
