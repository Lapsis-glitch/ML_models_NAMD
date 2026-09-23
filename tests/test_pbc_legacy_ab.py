"""A/B: the new cell-aware wrapper with a zero cell must reproduce the
archived pre-PBC artifact exactly.  This is the promise the archive README
makes, so it is worth asserting rather than assuming."""
import sys, torch
sys.path.insert(0, "/home/rat/PycharmProjects/ML_models_NAMD")
from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper

ARCHIVED = "models/archive_pre_pbc_2026-07-31/mace_off23_wrapped_identical.pt"
INNER    = "models/compiled_mace_off23_medium.pt"

old = torch.jit.load(ARCHIVED, map_location="cpu"); old.eval()
new = torch.jit.script(MACE_TS_Wrapper(INNER, device="cpu"))

print("archived forward args:", len(old.forward.schema.arguments))
print("new      forward args:", len(new.forward.schema.arguments))

torch.manual_seed(3)
c = torch.tensor([[0.,0.,0.],[0.96,0.,0.],[-0.24,0.93,0.],
                  [3.1,0.2,0.1],[4.06,0.2,0.1],[2.86,1.13,0.1]], dtype=torch.float64)
c = c + torch.randn_like(c) * 0.02
Z = torch.tensor([8,1,1,8,1,1], dtype=torch.int64)
pcx = torch.zeros((0,3),dtype=torch.float64); pcq = torch.zeros((0,),dtype=torch.float64)

eo, fo, qo = old(c, Z, pcx, pcq)
en, fn, qn, _ = new(c, Z, pcx, pcq, torch.zeros((1,3,3), dtype=torch.float64))

de = float((eo - en).abs().max()); df = float((fo - fn).abs().max())
print("energy  |diff| = %.3e" % de)
print("forces  |diff| = %.3e" % df)
exact = bool(torch.equal(eo, en) and torch.equal(fo, fn))
print("BIT-IDENTICAL:", exact)
ok = de == 0.0 and df == 0.0
print("RESULT:", "PASS" if ok else "FAIL")
if __name__ == "__main__":
    sys.exit(0 if ok else 1)
else:
    def test_legacy_bit_identical():
        """Collected by pytest; the comparison above runs at import."""
        assert ok, "zero-cell result differs from the archived artifact"
