from .hawkes import FIMHawkes, FIMHawkesConfig
from .hawkes_intensity_free import FIMHawkes as FIMHawkesIntensityFree
from .hawkes_intensity_free import FIMHawkesConfig as FIMHawkesIntensityFreeConfig


# Register with transformers for proper AutoConfig/AutoModel support
FIMHawkesConfig.register_for_auto_class()
FIMHawkes.register_for_auto_class("AutoModel")

FIMHawkesIntensityFreeConfig.register_for_auto_class()
FIMHawkesIntensityFree.register_for_auto_class("AutoModel")

__all__ = ["FIMHawkes", "FIMHawkesConfig", "FIMHawkesIntensityFree", "FIMHawkesIntensityFreeConfig"]
