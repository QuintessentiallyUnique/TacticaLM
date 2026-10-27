from trl import SFTConfig, SFTTrainer
import inspect

print("SFTConfig fields:")
if hasattr(SFTConfig, "__dataclass_fields__"):
    for k in SFTConfig.__dataclass_fields__.keys():
        if "seq" in k or "len" in k or "max" in k:
            print(k)

print("\nSFTTrainer args:")
for k in inspect.signature(SFTTrainer.__init__).parameters.keys():
    print(k)
