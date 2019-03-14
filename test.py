from log_lin_connect import *
import torch
x = torch.autograd.Variable(torch.FloatTensor([[1.1],[2]]), requires_grad = True)
y = torch.FloatTensor([[2],[2]])
lin = LinearQuant(1,1)

z = lin(x)
t = LogQuant(bitwight=16)(torch.nn.ReLU()(z))
print(z)
print(t.data)
print(lin.weight)
#opti = torch.optim.SGD([x], lr=0.1,momentum=0.0)



"""
for i in range(5):
    opti.zero_grad()
    
    res = torch.pow(LogQuant()(x)*y  - 16,2.0)
    print("res ", res.data)
    print("x " ,LogQuant()(x).item())
    res.backward()
    opti.step()
"""