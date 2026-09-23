import sys, torch
sys.path.insert(0, "/home/rat/PycharmProjects/ML_models_NAMD")
from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper
w = MACE_TS_Wrapper("models/compiled_mace_off23_medium.pt", device="cpu")
s = torch.jit.script(w)
L = 12.0                      # > 2*r_max = 10, so minimum image is valid
# Two waters that are FAR apart directly but ADJACENT through the x boundary.
c = torch.tensor([[0.4,6.,6.],[1.36,6.,6.],[0.16,6.93,6.],
                  [L-1.4,6.,6.],[L-0.44,6.,6.],[L-1.64,6.93,6.]], dtype=torch.float64)
Z = torch.tensor([8,1,1,8,1,1], dtype=torch.int64)
pcx = torch.zeros((0,3),dtype=torch.float64); pcq=torch.zeros((0,),dtype=torch.float64)
print("direct O-O distance      : %.3f A" % float((c[0]-c[3]).norm()))
print("through-boundary O-O dist: %.3f A" % float((c[0]-(c[3]-torch.tensor([L,0.,0.],dtype=torch.float64))).norm()))
e0,f0,_,_ = s(c,Z,pcx,pcq, torch.zeros((1,3,3),dtype=torch.float64))
eP,fP,_,_ = s(c,Z,pcx,pcq,(torch.eye(3,dtype=torch.float64)*L).unsqueeze(0))
print("non-periodic E = %.6f kcal/mol" % float(e0))
print("periodic     E = %.6f kcal/mol" % float(eP))
print("dE (interaction seen only with PBC) = %.6f kcal/mol" % float(eP-e0))
print("max force change = %.6f kcal/mol/A" % float((fP-f0).abs().max()))
# Reference: replicate the neighbour explicitly into the cluster, no PBC.
cref = c.clone(); cref[3:] -= torch.tensor([L,0.,0.],dtype=torch.float64)
eR,fR,_,_ = s(cref,Z,pcx,pcq, torch.zeros((1,3,3),dtype=torch.float64))
print("explicitly-imaged cluster E = %.6f kcal/mol" % float(eR))
print("PBC == explicit image  :", bool(torch.allclose(eP,eR,atol=1e-6)),
      " |dE| = %.2e" % float((eP-eR).abs()))
print("forces match too       :", bool(torch.allclose(fP,fR,atol=1e-6)),
      " max|df| = %.2e" % float((fP-fR).abs().max()))
